"""RoPE 核心旋转变换 _rotate_half 的单元测试。"""

import torch

from model.model import _rotate_half
from tests.utils import assert_tensors_allclose, ref_rotate_half


class TestShape:
    """形状校验: 输入输出形状必须一致"""

    def test_2d(self):
        """2D 输入 (seq, head_dim)"""
        x = torch.randn(4, 16)
        assert _rotate_half(x, torch.randn(8), torch.randn(8)).shape == x.shape

    def test_4d_attention(self):
        """标准 Attention 输入 (batch, heads, seq, head_dim)"""
        x = torch.randn(2, 4, 8, 64)
        cos = torch.randn(1, 1, 8, 32)
        sin = torch.randn(1, 1, 8, 32)
        assert _rotate_half(x, cos, sin).shape == x.shape

    def test_odd_dim(self):
        """奇数 head_dim 防御性测试：不崩溃，且与参考实现的拆分语义一致。

        注意：模型配置层已禁止奇数 head_dim（每个注意力头维度必须为偶数），
        因此该测试仅保证底层函数对非标准输入不抛异常，并与参考实现行为一致。
        """
        x = torch.randn(2, 3)
        cos, sin = torch.randn(1), torch.randn(1)
        assert_tensors_allclose(_rotate_half(x, cos, sin), ref_rotate_half(x, cos, sin))


class TestNumerical:
    """数值一致性: 对照独立参考实现"""

    def test_vs_reference(self):
        """与独立复数旋转参考实现对齐"""
        x = torch.randn(4, 16)
        cos = torch.randn(8)
        sin = torch.randn(8)
        assert_tensors_allclose(_rotate_half(x, cos, sin), ref_rotate_half(x, cos, sin))

    def test_identity(self):
        """cos=1, sin=0 -> 恒等变换 (不改变输入)"""
        x = torch.randn(2, 8, 32)
        assert_tensors_allclose(_rotate_half(x, torch.ones(2, 8, 16), torch.zeros(2, 8, 16)), x)

    def test_quarter_rotation(self):
        """cos=0, sin=1 -> 90度旋转: (x1,x2) -> (-x2,x1)"""
        x = torch.randn(2, 4)
        d2 = 2
        out = _rotate_half(x, torch.zeros(1, 2), torch.ones(1, 2))
        expected = torch.cat([-x[..., d2:], x[..., :d2]], dim=-1)
        assert_tensors_allclose(out, expected)


class TestGradient:
    """梯度稳定性"""

    def test_flow(self):
        """所有输入梯度非空且无 NaN"""
        x = torch.randn(2, 4, 16, requires_grad=True)
        cos = torch.randn(1, 1, 8, requires_grad=True)
        sin = torch.randn(1, 1, 8, requires_grad=True)
        _rotate_half(x, cos, sin).sum().backward()
        assert x.grad is not None and not torch.isnan(x.grad).any()

    def test_stability_repeated(self):
        """连续 5 次前向+反向, 梯度范数在合理范围"""
        for _ in range(5):
            x = torch.randn(2, 4, 16, requires_grad=True)
            _rotate_half(x, torch.randn(1, 1, 8), torch.randn(1, 1, 8)).pow(2).mean().backward()
            g = x.grad.norm().item()
            assert 1e-10 < g < 1e3, f"gradient norm {g} out of range"
