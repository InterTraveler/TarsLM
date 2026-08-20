"""RMSNorm 的单元测试。"""

import torch

from model.model import TarsLMRMSNorm
from tests.utils import assert_tensors_allclose, ref_rms_norm


class TestShape:
    """形状校验"""

    def test_output(self):
        """输入 (batch,seq,hidden) 输出形状相同"""
        assert TarsLMRMSNorm(64)(torch.randn(2, 8, 64)).shape == (2, 8, 64)

    def test_batch_independent(self):
        """不同 batch size 均正常"""
        n = TarsLMRMSNorm(64)
        for b in [1, 2, 4]: assert n(torch.randn(b, 16, 64)).shape == (b, 16, 64)


class TestNumerical:
    """数值一致性"""

    def test_vs_reference(self):
        """与独立公式参考对齐"""
        n = TarsLMRMSNorm(64, eps=1e-6)
        x = torch.randn(2, 8, 64)
        assert_tensors_allclose(n(x), ref_rms_norm(x, n.weight, n.eps))

    def test_scale_invariance(self):
        """输入整体缩放不改变输出（仅当 gamma 全为 1 时成立）。

        该性质源于归一化本身消除量纲：rms(10x) = 10 * rms(x)。
        注意：可学习权重 gamma 不为全 1 时此性质不再成立，
        因此本测试只作用于初始化后的模块（权重初始为全 1）。
        """
        n = TarsLMRMSNorm(64, eps=1e-8)
        x = torch.randn(4, 8, 64)
        assert_tensors_allclose(n(x), n(x * 10), rtol=1e-4)

    def test_zero_input(self):
        """全零输入不产生 NaN/Inf (eps 防除零)"""
        out = TarsLMRMSNorm(64)(torch.zeros(2, 8, 64))
        assert not torch.isnan(out).any() and not torch.isinf(out).any()

    def test_unit_rms(self):
        """gamma=1 时输出 RMS 应接近 1"""
        n = TarsLMRMSNorm(64, eps=1e-8)
        n.weight.data.fill_(1.0)
        rms = torch.sqrt(n(torch.randn(2, 8, 64)).pow(2).mean(-1))
        assert torch.allclose(rms, torch.ones_like(rms), rtol=1e-3)


class TestInit:
    """初始化校验"""

    def test_weight_ones(self):
        """gamma 初始值为全 1 (恒等映射)"""
        n = TarsLMRMSNorm(64)
        assert_tensors_allclose(n.weight, torch.ones(64))


class TestGradient:
    """梯度稳定性"""

    def test_flow(self):
        """输入和 weight 都有有效梯度"""
        n = TarsLMRMSNorm(64)
        x = torch.randn(2, 8, 64, requires_grad=True)
        n(x).sum().backward()
        assert x.grad is not None and n.weight.grad is not None

    def test_large_values_stable(self):
        """大数值输入 (x100) 梯度不消失"""
        n = TarsLMRMSNorm(64)
        x = torch.randn(2, 8, 64).mul_(100).requires_grad_(True)
        n(x).pow(2).mean().backward()
        assert 1e-12 < x.grad.norm().item() < 1e3
