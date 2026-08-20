"""TarsLMRotaryEmbedding 旋转位置编码的单元测试。"""

import torch

from model.config import TarsLMConfig
from model.model import TarsLMRotaryEmbedding
from tests.utils import RefRotaryEmbedding, assert_tensors_allclose


class TestShape:
    """形状校验"""

    def test_output(self):
        """Q/K 输出形状与输入一致"""
        r = TarsLMRotaryEmbedding(dim=32, max_seq_len=128)
        x = torch.randn(2, 4, 16, 32)
        q, k = r(x, x)
        assert q.shape == x.shape and k.shape == x.shape

    def test_variable_seq_len(self):
        """不同序列长度 (1,4,16) 均正常"""
        r = TarsLMRotaryEmbedding(dim=32, max_seq_len=128)
        for s in [1, 4, 16]:
            x = torch.randn(2, 4, s, 32)
            q, k = r(x, x)
            assert q.shape == x.shape and k.shape == x.shape


class TestNumerical:
    """数值一致性"""

    def test_vs_reference(self):
        """与独立 RoPE 编码器参考对齐"""
        r1 = TarsLMRotaryEmbedding(dim=32, max_seq_len=64)
        r2 = RefRotaryEmbedding(dim=32, max_seq_len=64)
        r2.cos.copy_(r1.cos)
        r2.sin.copy_(r1.sin)
        x = torch.randn(2, 4, 16, 32)
        q1, k1 = r1(x, x)
        q2, k2 = r2(x, x)
        assert_tensors_allclose(q1, q2)
        assert_tensors_allclose(k1, k2)

    def test_position_sensitivity(self):
        """不同位置的相同 token 产生不同编码 (位置信息生效)"""
        r = TarsLMRotaryEmbedding(dim=32, max_seq_len=128)
        x = torch.randn(1, 1, 2, 32)
        q, _ = r(x, x)
        assert (q[0, 0, 0] - q[0, 0, 1]).abs().sum() > 1e-4

    def test_identity_when_cos1_sin0(self):
        """cos=1, sin=0 时退化为恒等变换"""
        r = TarsLMRotaryEmbedding(dim=32, max_seq_len=128)
        r.cos[:] = 1.0
        r.sin[:] = 0.0
        x = torch.randn(2, 4, 8, 32)
        q, _ = r(x, x)
        assert_tensors_allclose(q, x)


class TestGradient:
    """梯度与参数属性"""

    def test_flow(self):
        """Q/K 梯度存在且无 NaN"""
        r = TarsLMRotaryEmbedding(dim=32, max_seq_len=128)
        xq = torch.randn(2, 4, 8, 32, requires_grad=True)
        xk = torch.randn(2, 4, 8, 32, requires_grad=True)
        q, k = r(xq, xk)
        (q.sum() + k.sum()).backward()
        assert xq.grad is not None and xk.grad is not None

    def test_cos_sin_are_buffers(self):
        """cos/sin 是 buffer, 不参与梯度更新"""
        r = TarsLMRotaryEmbedding(dim=32, max_seq_len=128)
        assert not r.cos.requires_grad and not r.sin.requires_grad


class TestRelativePosition:
    """RoPE 核心数学性质：相对位置不变性。

    旋转位置编码的关键特征：位置 i 的 query 与位置 j 的 key 的点积，
    仅取决于相对距离 delta = i - j，与绝对位置无关。违反此性质意味着
    RoPE 实现有 bug，模型无法泛化到训练时未见过的序列长度。
    """

    def test_relative_invariance(self):
        """位置 i,j 的 QK 点积 == 位置 i+k,j+k 的 QK 点积。

        静默 bug：如果 RoPE 使用了绝对位置而非相对旋转，此不变性会崩溃，
        导致模型无法泛化到新序列长度。将同一个特征向量复制到所有位置，
        验证任意两个位置的 QK 点积仅取决于它们的相对距离。
        """
        dim = 32
        r = TarsLMRotaryEmbedding(dim=dim, max_seq_len=64)
        # 把同一个特征向量复制到所有位置（位置 0 和位置 4 内容相同）
        vec = torch.randn(1, 1, 1, dim).expand(1, 1, 8, dim)
        x_q = vec.clone()
        x_k = vec.clone()
        q, k = r(x_q, x_k)
        # 计算所有位置对的 QK 点积矩阵 dot[i][j]
        dot = (q[0, 0] @ k[0, 0].T)  # (8, 8)
        # 相对位置 (i-j) 决定点积大小，因此 dot[i, j] ~ dot[i+1, j+1]
        diffs = dot[:-1, :-1] - dot[1:, 1:]
        assert diffs.abs().max().item() < 1e-5, (
            f"RoPE 相对位置不变性被破坏: 最大偏差={diffs.abs().max().item():.2e}"
        )


class TestVNotRotated:
    """验证 RoPE 只旋转 Q 和 K，不旋转 V。

    RoPE 的设计原则：位置信息只影响"谁关注谁"（Attention Score），
    不影响"关注的内容是什么"（Attention Output）。如果错误地对 V 也施加
    RoPE 旋转，会导致计算浪费且可能降低注意力质量。
    """

    def test_v_not_rotated(self):
        """Q/K 被旋转，且 rotary_emb 的调用入参中从不出现 V。

        通过 spy 包装 rotary_emb.forward，记录模型前向中旋转函数的实际调用：
        必须恰好一次、且入参只有 Q 和 K 两个张量。若 Attention 内部错误地
        把 V 也传入 RoPE，会出现第二次调用或入参中出现 V，测试即失败。
        """
        import torch.nn.functional as F

        from model.attention import TarsLMAttention
        cfg = TarsLMConfig(hidden_size=64, num_attention_heads=4,
                           num_key_value_heads=4, max_seq_len=128, attention_dropout=0.0,
                           hidden_dropout=0.0, rope_theta=10000.0)
        attn = TarsLMAttention(cfg)
        x = torch.randn(1, 4, 64)

        # 手动提取 Q/K 投影后的未旋转结果，用于证明 RoPE 确实生效
        q_unrot = F.linear(x, attn.q_proj.weight).view(1, 4, 4, 16).transpose(1, 2)
        k_unrot = F.linear(x, attn.k_proj.weight).view(1, 4, 4, 16).transpose(1, 2)

        # spy 包装：记录 rotary_emb 前向的实际入参与返回值
        calls = []
        original_forward = attn.rotary_emb.forward

        def spy_forward(xq, xk, *args, **kwargs):
            q_out, k_out = original_forward(xq, xk, *args, **kwargs)
            calls.append((xq, xk, q_out, k_out))
            return q_out, k_out

        attn.rotary_emb.forward = spy_forward
        attn(x)

        # 旋转只发生一次，且入参只有 Q 和 K（V 从未进入旋转路径）
        assert len(calls) == 1, "rotary_emb 应只被调用一次（Q、K 各一份），V 不应被旋转"
        _, _, q_rot, k_rot = calls[0]
        # Q 和 K 必须与未旋转版本不同，证明 RoPE 生效
        assert not torch.allclose(q_unrot, q_rot, atol=1e-5), "Q 未被 RoPE 旋转"
        assert not torch.allclose(k_unrot, k_rot, atol=1e-5), "K 未被 RoPE 旋转"
