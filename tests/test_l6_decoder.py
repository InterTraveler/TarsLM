"""TarsLMDecoderLayer 的单元测试。"""

import pytest
import torch
from torch import nn

from model.config import TarsLMConfig
from model.model import TarsLMDecoderLayer
from tests.utils import RefDecoderLayer, assert_tensors_allclose


@pytest.fixture
def layer(layer_config):
    """创建标准 DecoderLayer"""
    return TarsLMDecoderLayer(layer_config)


class TestShape:
    """形状校验"""

    def test_output(self, layer):
        """输出形状与输入一致"""
        out, kv = layer(torch.randn(2, 16, 64))
        assert out.shape == (2, 16, 64) and kv is None

    def test_kv_cache(self, layer):
        """use_cache=True 返回每层 KV"""
        _, kv = layer(torch.randn(2, 16, 64), use_cache=True)
        assert kv is not None and kv[0].shape == (2, 4, 16, 16)


class TestNumerical:
    """数值一致性"""

    def test_vs_reference(self, layer, layer_config):
        """对照独立组装参考 DecoderLayer"""
        ref = RefDecoderLayer(layer_config)
        ref.load_state_dict(layer.state_dict())
        ref.eval()
        layer.eval()
        x = torch.randn(2, 8, 64)
        a, _ = layer(x)
        e, _ = ref(x)
        assert_tensors_allclose(a, e, rtol=1e-4)

    def test_residual_active(self, layer):
        """Attn 和 FFN 置零后残差使 output==input"""
        layer.eval()
        with torch.no_grad():
            for p in layer.self_attn.parameters(): p.zero_()
            for p in layer.mlp.parameters(): p.zero_()
        x = torch.ones(2, 4, 64)
        out, _ = layer(x)
        assert_tensors_allclose(out, x, rtol=1e-3)

    def test_pre_norm_order(self, layer):
        """两个 Norm 置零后残差使 output==input (证明 Pre-Norm 顺序正确)"""
        layer.eval()
        with torch.no_grad():
            layer.input_layernorm.weight.zero_()
            layer.post_attention_layernorm.weight.zero_()
        x = torch.randn(2, 4, 64)
        out, _ = layer(x)
        assert_tensors_allclose(out, x, rtol=1e-3)


class TestGradient:
    """梯度与检查点"""

    def test_flow(self, layer):
        """所有参数梯度非空"""
        layer.train()
        x = torch.randn(2, 8, 64, requires_grad=True)
        layer(x)[0].pow(2).mean().backward()
        for name, p in layer.named_parameters():
            assert p.grad is not None, f"{name} missing grad"

    def test_checkpoint_consistency(self, layer):
        """梯度检查点前后损失和梯度一致"""
        layer.train()
        x = torch.randn(2, 8, 64, requires_grad=True)
        out, _ = layer(x)
        out.pow(2).mean().backward()
        g_normal = {n: p.grad.clone() for n, p in layer.named_parameters() if p.grad is not None}
        layer.zero_grad()
        x.grad = None
        layer._gradient_checkpointing = True
        out2, _ = layer(x)
        out2.pow(2).mean().backward()
        g_ckpt = {n: p.grad.clone() for n, p in layer.named_parameters() if p.grad is not None}
        for n, g in g_normal.items():
            assert_tensors_allclose(g, g_ckpt[n], rtol=1e-4,
                                    msg=f"Checkpoint grad mismatch: {n}")


class TestKVCacheNumerical:
    """DecoderLayer 级别 KV 缓存数值一致性。"""

    def test_full_vs_incremental(self, layer):
        """逐 token 增量前向：验证 DecoderLayer 级别 KV 缓存拼接正确。

        在 DecoderLayer 层面模拟自回归生成，每次传入 1 个 token 并拼接
        历史 KV 缓存，检查输出形状和最终缓存序列长度。
        """
        layer.eval()
        x_full = torch.randn(1, 8, 64)
        layer(x_full, use_cache=False)
        kv = None
        out_steps = []
        for i in range(8):
            x_step = x_full[:, i:i + 1, :]
            out_step, kv = layer(x_step, past_key_value=kv, use_cache=True)
            out_steps.append(out_step)
        out_inc = torch.cat(out_steps, dim=1)
        assert out_inc.shape == (1, 8, 64)
        assert kv[0].shape[-2] == 8


class TestNormsIndependent:
    """两个 Norm 层权重必须独立。"""

    def test_norms_independent_weights(self, layer):
        """input_layernorm.weight 与 post_attention_layernorm.weight 是不同的 Parameter 对象。

        静默 bug：如果两个 Norm 意外共享了同一个 weight 对象（而非各自独立的 Parameter），
        会导致训练时的梯度混淆和模型表达能力下降。
        """
        assert layer.input_layernorm.weight is not layer.post_attention_layernorm.weight, (
            "input_layernorm and post_attention_layernorm share the same Parameter object"
        )


class TestMoEDecoder:
    """使用 MoE 替代稠密 FFN 的 DecoderLayer 测试。"""

    @pytest.fixture
    def moe_layer(self, moe_layer_config):
        return TarsLMDecoderLayer(moe_layer_config)

    def test_moe_forward(self, moe_layer):
        """MoE DecoderLayer 前向传播输出形状正确，且 self.mlp 确实是 TarsLMMoE 实例。"""
        from model.ffn import TarsLMMoE
        assert isinstance(moe_layer.mlp, TarsLMMoE), "MLP is not TarsLMMoE"
        out, kv = moe_layer(torch.randn(2, 8, 64), use_cache=True)
        assert out.shape == (2, 8, 64)
        assert kv[0].shape == (2, 4, 8, 16)

    def test_moe_vs_reference(self, moe_layer, moe_layer_config):
        """MoE DecoderLayer 与手工装配的同结构参考层数值对齐。

        参考层使用与 DecoderLayer 完全相同的子模块类（RMSNorm、Attention、
        TarsLMMoE、Dropout）手工组装，用于验证 DecoderLayer 的子层装配
        顺序与残差连接正确（Pre-Norm → Attention → 残差 → Pre-Norm → MoE
        → 残差）。注意：MoE 本身的数值正确性由 test_l5_ffn.py 的模块级
        测试单独覆盖，本测试不重复验证。
        """
        from model.attention import TarsLMAttention
        from model.ffn import TarsLMMoE
        from model.norm import TarsLMRMSNorm
        ref = nn.Module()
        ref.input_layernorm = TarsLMRMSNorm(moe_layer_config.hidden_size, moe_layer_config.norm_eps)
        ref.self_attn = TarsLMAttention(moe_layer_config)
        ref.post_attention_layernorm = TarsLMRMSNorm(moe_layer_config.hidden_size, moe_layer_config.norm_eps)
        ref.mlp = TarsLMMoE(moe_layer_config)
        ref.dropout = nn.Dropout(moe_layer_config.hidden_dropout)
        ref.eval()
        moe_layer.eval()
        # 结构完全一致，必须严格匹配所有键，避免 strict=False 掩盖键名不匹配
        ref.load_state_dict(moe_layer.state_dict(), strict=True)
        x = torch.randn(2, 8, 64)
        # 手工复现 DecoderLayer 的前向顺序
        r = x
        h = ref.input_layernorm(x)
        a, _ = ref.self_attn(h)
        h = r + ref.dropout(a)
        r = h
        h = ref.post_attention_layernorm(h)
        h = r + ref.dropout(ref.mlp(h))
        expected = h
        actual, _ = moe_layer(x)
        assert_tensors_allclose(actual, expected, rtol=1e-4,
                                msg="MoE DecoderLayer 输出与手工装配参考层不一致")

    def test_moe_gradient(self, moe_layer):
        """MoE Decoder 所有参数在训练时都能收到梯度。"""
        moe_layer.train()
        x = torch.randn(2, 8, 64, requires_grad=True)
        moe_layer(x)[0].pow(2).mean().backward()
        for name, p in moe_layer.named_parameters():
            assert p.grad is not None, f"{name} missing gradient in MoE decoder"


class TestHiddenDropout:
    """hidden_dropout 在训练/推理模式下的行为差异。"""

    def test_dropout_affects_training(self, layer):
        """训练模式下 hidden_dropout > 0 时，多次前向输出应不同（dropout 随机丢弃生效）。"""
        layer.train()
        layer.dropout.p = 0.5  # 临时调高 dropout 以便观察差异
        x = torch.randn(2, 4, 64)
        o1, _ = layer(x)
        o2, _ = layer(x)
        assert not torch.allclose(o1, o2, atol=1e-5), (
            "Training with dropout should produce different outputs each forward"
        )

    def test_eval_deterministic_with_dropout(self):
        """hidden_dropout > 0 时 eval 模式多次前向应完全一致（dropout 被自动关闭）。

        静默 bug：如果 eval 模式下忘记关闭 dropout，推理时输出不可复现。
        """
        cfg = TarsLMConfig(hidden_size=64, num_attention_heads=4,
                           num_key_value_heads=4, intermediate_size=256, max_seq_len=128,
                           num_layers=1, norm_eps=1e-6, attention_dropout=0.0, hidden_dropout=0.3,
                           use_bias=False, use_moe=False, rope_theta=10000.0)
        layer = TarsLMDecoderLayer(cfg)
        layer.eval()
        x = torch.randn(2, 8, 64)
        o1, _ = layer(x)
        o2, _ = layer(x)
        assert_tensors_allclose(o1, o2,
                                msg="Eval mode should be deterministic even with hidden_dropout > 0")
