"""多头自注意力 TarsLMAttention 的单元测试。"""

import pytest
import torch
import torch.nn.functional as F

from model.config import TarsLMConfig
from model.model import TarsLMAttention
from tests.utils import assert_tensors_allclose, ref_multi_head_attention


@pytest.fixture
def attn(attn_config):
    """创建标准 Attention 模块"""
    return TarsLMAttention(attn_config)


class TestShape:
    """形状校验"""

    def test_output(self, attn):
        """输出形状与输入一致, use_cache=False 时 kv=None"""
        out, kv = attn(torch.randn(2, 16, 64))
        assert out.shape == (2, 16, 64) and kv is None

    def test_kv_cache_shape(self, attn):
        """use_cache=True 返回 (k,v), 形状 (batch,heads,seq,head_dim)"""
        _, kv = attn(torch.randn(2, 16, 64), use_cache=True)
        assert kv is not None and kv[0].shape == (2, 4, 16, 16)

    def test_past_kv_concat_len(self, attn):
        """增量推理: 拼接历史 KV 后序列长度 4->5"""
        _, kv1 = attn(torch.randn(2, 4, 64), use_cache=True)
        _, kv2 = attn(torch.randn(2, 1, 64), past_key_value=kv1, use_cache=True)
        assert kv2[0].shape[-2] == 5


class TestCausal:
    """因果注意力: token i 不能看到 token j (j > i)"""

    def test_no_future_leak(self, attn):
        """修改位置 4+ 的 token 不影响位置 0-3 的输出"""
        attn.eval()
        x = torch.zeros(2, 8, 64)
        x[:, :4, :] = 1.0
        out_normal, _ = attn(x)
        x_poisoned = x.clone()
        x_poisoned[:, 4:, :] = 100.0
        out_poisoned, _ = attn(x_poisoned)
        assert (out_normal[:, :4] - out_poisoned[:, :4]).abs().max().item() < 1e-4


class TestMask:
    """注意力掩码"""

    def test_padding_mask_suppresses(self, attn):
        """padding mask=0 的位置被完全屏蔽，其内容不影响后续有效 token。

        回归测试（修复 attention_mask additive 转换 bug）：把位于有效 token
        因果视野内的位置 3 屏蔽并投毒，验证位置 4+ 的输出不受影响。
        注意：投毒位置必须位于被断言位置的因果视野内（位置 4 可以关注到
        位置 3），否则因果掩码本身就会掩盖 padding mask 的缺陷。
        """
        attn.eval()
        x = torch.randn(2, 8, 64)
        mask = torch.ones(2, 8)
        mask[:, 3] = 0.0
        out_normal, _ = attn(x, attention_mask=mask)
        x_poisoned = x.clone()
        x_poisoned[:, 3, :] = 1e6
        out_poisoned, _ = attn(x_poisoned, attention_mask=mask)
        # 位置 4+ 会关注到位置 3，因此这里才能真正检验 padding mask 是否生效
        assert_tensors_allclose(out_normal[:, 4:], out_poisoned[:, 4:], rtol=1e-3,
                                msg="padding mask 未屏蔽被投毒的位置 3")

    def test_2d_mask_does_not_change_valid_tokens(self, attn):
        """因果掩码正确时，padding mask 不应改变有效 token 的输出。"""
        attn.eval()
        x = torch.randn(2, 8, 64)
        out1, _ = attn(x)
        mask = torch.ones(2, 8)
        mask[:, 4:] = 0.0
        out2, _ = attn(x, attention_mask=mask)
        assert_tensors_allclose(out1[:, :4], out2[:, :4], rtol=1e-3,
                                msg="padding mask 不应改变因果遮蔽下的有效 token 输出")

    def test_causal_still_works_with_padding_mask(self, attn):
        """padding mask 存在时因果掩码仍生效：位置 i 不能看到位置 j (j>i)。
        复现条件：HF Trainer 训练时始终传递 attention_mask（即使全 1），
        若 is_causal 错误设为 False，此测试会失败。
        """
        attn.eval()
        x = torch.randn(2, 8, 64)
        mask = torch.ones(2, 8)  # 模拟 Trainer 传入的全 1 mask
        out_normal, _ = attn(x, attention_mask=mask)
        x_poisoned = x.clone()
        x_poisoned[:, 5:, :] = 1e6
        out_poisoned, _ = attn(x_poisoned, attention_mask=mask)
        assert_tensors_allclose(out_normal[:, :4], out_poisoned[:, :4], rtol=1e-3,
                                msg="因果掩码在 padding mask 存在时被错误关闭")


class TestNumerical:
    """数值一致性: vs 手写 softmax attention"""

    def test_vs_manual(self, attn, attn_config):
        """屏蔽 RoPE 后对比 SDPA 与手写实现的差异"""
        attn.eval()
        x = torch.randn(2, 8, 64)
        q_w = attn.q_proj.weight
        k_w = attn.k_proj.weight
        v_w = attn.v_proj.weight
        o_w = attn.o_proj.weight
        bsz, seq = x.shape[:2]
        nh, hd = attn.num_heads, attn.head_dim
        q = F.linear(x, q_w).view(bsz, seq, nh, hd).transpose(1, 2)
        k = F.linear(x, k_w).view(bsz, seq, nh, hd).transpose(1, 2)
        v = F.linear(x, v_w).view(bsz, seq, nh, hd).transpose(1, 2)
        q, k = attn.rotary_emb(q, k)
        ref = ref_multi_head_attention(q, k, v, is_causal=True)
        ref = ref.transpose(1, 2).contiguous().view(bsz, seq, -1)
        ref = F.linear(ref, o_w)
        actual, _ = attn(x)
        assert_tensors_allclose(actual, ref, rtol=1e-3)


class TestGradient:
    """梯度与训练/推理一致性"""

    def test_all_params_gradient(self, attn):
        """训练时所有参数都有有效梯度"""
        attn.train()
        x = torch.randn(2, 8, 64, requires_grad=True)
        attn(x)[0].pow(2).mean().backward()
        for name, p in attn.named_parameters():
            assert p.grad is not None, f"{name} missing grad"

    def test_eval_deterministic(self, attn):
        """eval 模式多次 forward 输出完全一致 (dropout 关闭)"""
        attn.eval()
        x = torch.randn(2, 8, 64)
        out1, _ = attn(x)
        out2, _ = attn(x)
        assert_tensors_allclose(out1, out2)


class TestKVCacheNumerical:
    """KV 缓存数值一致性：增量推理拼接逻辑正确性。

    静默 bug：如果 KV 缓存的拼接顺序或维度处理有误，增量推理会产生与
    全量前向不同的静默错误输出。本测试逐 token 增量前向，验证缓存序列长度
    逐步增长、输出形状始终正确、无 NaN 崩溃。
    """

    def test_full_vs_incremental(self, attn):
        """逐 token 增量前向，验证 KV 缓存逐步拼接后序列长度和输出形状正确。

        模拟自回归生成流程：第一次前向 1 个 token，之后每次传入 1 个新 token
        并拼接历史 KV 缓存。检查 8 步后的输出拼接结果形状为 (1, 8, 64)，
        且最终 KV 缓存的序列长度为 8（而非泄漏或截断）。
        """
        attn.eval()
        x_full = torch.randn(1, 8, 64)
        attn(x_full, use_cache=False)
        # 增量模式：逐 token 前向，每次传入 x_full 的一个位置
        kv = None
        out_steps = []
        for i in range(8):
            x_step = x_full[:, i:i + 1, :]
            out_step, kv = attn(x_step, past_key_value=kv, use_cache=True)
            out_steps.append(out_step)
        out_inc = torch.cat(out_steps, dim=1)
        # 增量输出拼接后形状应与全量输出一致
        assert out_inc.shape == (1, 8, 64)
        assert kv[0].shape[-2] == 8, f"Cache seq len should be 8, got {kv[0].shape[-2]}"

    def test_full_vs_prefix_then_batch(self, attn):
        """前缀缓存 + 批量续推：即使 use_cache=False，也必须复用历史 KV。"""
        attn.eval()
        x_full = torch.randn(1, 8, 64)
        out_full, _ = attn(x_full, use_cache=False)
        # 前缀：前 4 个 token，带 KV 缓存
        out_prefix, kv = attn(
            x_full[:, :4, :],
            use_cache=True,
            position_ids=torch.arange(4).unsqueeze(0),
        )
        # 批量续推：后 4 个 token，使用前缀的 KV 缓存
        out_batch, _ = attn(
            x_full[:, 4:, :],
            past_key_value=kv,
            use_cache=False,
            position_ids=torch.arange(4, 8).unsqueeze(0),
        )
        out_combined = torch.cat([out_prefix, out_batch], dim=1)
        assert_tensors_allclose(
            out_combined,
            out_full,
            rtol=1e-4,
            atol=1e-5,
            msg="use_cache=False 时历史 KV 缓存被错误忽略",
        )


class TestGQA:
    """GQA（Grouped Query Attention）分组查询正确性。

    GQA 核心操作 _repeat_kv 将 num_kv_heads 个 KV 头沿组维度复制展开到
    num_heads 个头。静默 bug：维度错位导致形状正确但数值错误（例如把不同
    KV 头的内容混入了错误的 Q 头组中）。
    """

    def test_repeat_kv_head_count(self, gqa_config):
        """_repeat_kv 展开后 KV 头数必须等于 Q 头数。

        用 num_heads=8, num_kv_heads=2 创建 GQA Attention，手动计算 K/V 投影。
        验证 _repeat_kv 将 (2, 2, seq, dim) 正确展开为 (2, 8, seq, dim)。
        """
        attn = TarsLMAttention(gqa_config)
        x = torch.randn(2, 8, 64)
        k = attn.k_proj(x).view(2, 8, 2, 8).transpose(1, 2)  # (2, 2, 8, 8)
        v = attn.v_proj(x).view(2, 8, 2, 8).transpose(1, 2)  # (2, 2, 8, 8)
        k_expanded = attn._repeat_kv(k)
        v_expanded = attn._repeat_kv(v)
        assert k_expanded.shape[1] == 8, f"Expected 8 heads after repeat, got {k_expanded.shape[1]}"
        assert v_expanded.shape[1] == 8

    def test_gqa_group_consistency(self, gqa_config):
        """同一 KV 组内展开的头必须完全相同，不同 KV 组必须不同。

        num_kv_heads=2, group_size=4 时：头 0-3 来自 KV 头 0 的复制（应相等），
        头 4-7 来自 KV 头 1 的复制（应相等），但头 0 与头 4 必须不同。
        静默 bug：_repeat_kv 中 expand/reshape 的维度顺序错位会导致跨组混淆。
        """
        attn = TarsLMAttention(gqa_config)
        x = torch.randn(2, 8, 64)
        k = attn.k_proj(x).view(2, 8, 2, 8).transpose(1, 2)
        k_expanded = attn._repeat_kv(k)
        # 头 0 与头 1 应完全一致（同属 KV 头 0 组，组内复制）
        assert_tensors_allclose(k_expanded[:, 0], k_expanded[:, 1],
                                msg="GQA repeat: adjacent heads in same group differ")
        assert_tensors_allclose(k_expanded[:, 0], k_expanded[:, 3],
                                msg="GQA repeat: first and last head in same group differ")
        # 头 0 与头 4 应不同（分属不同 KV 头组）
        assert not torch.allclose(k_expanded[:, 0], k_expanded[:, 4], atol=1e-5), (
            "GQA repeat: heads from different KV heads should differ")

    def test_gqa_forward_output_shape(self, gqa_config):
        """GQA 完整前向传播输出形状正确，且 KV 缓存存储未展开的头数。

        完整调用 TarsLMAttention.forward，验证输出形状 (batch, seq, hidden) 正确，
        且 KV 缓存中 K 的头数为 num_kv_heads=2（而非展开后的 8）。
        静默 bug：如果在缓存前就展开了 K/V，每次增量步骤会浪费 4x 显存。
        """
        attn = TarsLMAttention(gqa_config)
        out, kv = attn(torch.randn(2, 16, 64), use_cache=True)
        assert out.shape == (2, 16, 64)
        # KV 缓存应存储未展开的 num_kv_heads=2，而非展开后的 8
        assert kv[0].shape[1] == 2, f"KV cache should have 2 heads, got {kv[0].shape[1]}"


class TestEdgeCases:
    """边界场景：seq_len=1、batch_size=1、最小 head_dim 等。

    这些极端场景最容易触发维度错位、广播失败等静默 bug，
    因为大多数代码都按"正常"的 batch 和 seq 维度编写。
    """

    def test_seq_len_1(self, attn):
        """seq_len=1 边界：因果掩码 + 维度对齐不崩溃。

        seq_len=1 时 QK^T 为 1x1 矩阵，因果掩码为空操作。
        但错误的 reshape 可能将 (bsz, 1, hidden) 错误地当成 (bsz, 1, heads*dim)
        而触发维度不匹配。
        """
        attn.eval()
        out, kv = attn(torch.randn(2, 1, 64), use_cache=True)
        assert out.shape == (2, 1, 64)
        assert kv[0].shape == (2, 4, 1, 16)

    def test_batch_size_1(self, attn):
        """batch=1 边界：批量矩阵乘法的维度折叠正确性。

        batch=1 时如果错误地 squeeze 掉了 batch 维度，会导致
        (1, heads, seq, dim) 变成 (heads, seq, dim) 而后续 matmul 失败。
        """
        attn.eval()
        out, _ = attn(torch.randn(1, 16, 64))
        assert out.shape == (1, 16, 64)

    def test_head_dim_8(self):
        """最小 head_dim=8 边界：RoPE 频率计算需要 head_dim 能被 2 整除。

        head_dim=8 意味着 RoPE 后半维度为 4，频率表形状 (4,)。
        如果代码中硬编码了更大的 head_dim 假设，此处会崩溃。
        """
        cfg = TarsLMConfig(hidden_size=64, num_attention_heads=8,
                           num_key_value_heads=8, max_seq_len=64, attention_dropout=0.0,
                           rope_theta=10000.0)
        attn = TarsLMAttention(cfg)
        out, _ = attn(torch.randn(2, 8, 64))
        assert out.shape == (2, 8, 64)

    def test_mha_equals_gqa_with_identical_kv(self, attn_config):
        """MHA 模式（num_kv_heads == num_heads）输出必须与等价 GQA 一致。

        当 num_kv_heads == num_heads 时 _repeat_kv 为 no-op，GQA 退化为 MHA。
        将 MHA 的权重复制到 GQA 配置的 Attention 中，两者前向输出应完全相同。
        静默 bug：如果 K/V 投影维度计算依赖了错误的条件分支，
        MHA 和 GQA(no-op) 模式可能产生不同结果。
        """
        attn1 = TarsLMAttention(attn_config)  # 标准 MHA 模式
        gqa_cfg = TarsLMConfig(hidden_size=64, num_attention_heads=4,
                               num_key_value_heads=4, max_seq_len=128, attention_dropout=0.0,
                               hidden_dropout=0.0, rope_theta=10000.0)
        attn2 = TarsLMAttention(gqa_cfg)  # num_kv_heads==num_heads，退化为 MHA
        x = torch.randn(2, 8, 64)
        # 复制权重
        attn2.load_state_dict(attn1.state_dict())
        attn1.eval()
        attn2.eval()
        o1, _ = attn1(x)
        o2, _ = attn2(x)
        assert_tensors_allclose(o1, o2,
                                msg="MHA vs GQA(num_kv_heads==num_heads) should be identical")
