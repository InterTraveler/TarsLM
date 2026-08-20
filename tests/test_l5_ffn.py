"""SwiGLU 前馈网络 TarsLMFeedForward 的单元测试。"""

import pytest
import torch
import torch.nn.functional as F

from model.config import TarsLMConfig
from model.model import TarsLMFeedForward
from tests.utils import assert_tensors_allclose, ref_swiglu_ffn


@pytest.fixture
def ffn(ffn_config):
    """创建标准 SwiGLU FFN 模块"""
    return TarsLMFeedForward(ffn_config)


class TestShape:
    """形状校验: 输出等于输入"""

    def test_output(self, ffn):
        assert ffn(torch.randn(2, 16, 64)).shape == (2, 16, 64)

    def test_batch_independent(self, ffn):
        """每个 batch 独立计算, 分开算与一起算一致"""
        x = torch.randn(4, 8, 64)
        out = ffn(x)
        assert_tensors_allclose(out[0:1], ffn(x[0:1]))


class TestNumerical:
    """数值一致性"""

    def test_vs_reference(self, ffn):
        """与独立 SwiGLU 公式参考对齐"""
        x = torch.randn(2, 8, 64)
        expected = ref_swiglu_ffn(x, ffn.gate_proj.weight, ffn.up_proj.weight, ffn.down_proj.weight)
        assert_tensors_allclose(ffn(x), expected)

    def test_zero_weights_zero_output(self, ffn):
        """所有权重置零时输出全零 (无 bias 泄漏)"""
        for p in ffn.parameters(): p.data.zero_()
        assert_tensors_allclose(ffn(torch.randn(2, 4, 64)), torch.zeros(2, 4, 64))


class TestGradient:
    """梯度稳定性"""

    def test_all_params_flow(self, ffn):
        """所有权重参数都有梯度"""
        ffn.train()
        x = torch.randn(2, 8, 64, requires_grad=True)
        ffn(x).pow(2).mean().backward()
        for name, p in ffn.named_parameters():
            assert p.grad is not None, f"{name} missing grad"

    def test_stable_norm(self, ffn):
        """连续 3 次训练梯度范数在合理范围"""
        ffn.train()
        for _ in range(3):
            x = torch.randn(2, 8, 64, requires_grad=True)
            ffn(x).pow(2).mean().backward()
            g = sum(p.grad.norm().item() for p in ffn.parameters())
            assert 1e-10 < g < 1e3
            ffn.zero_grad()


class TestMoE:
    """TarsLMMoE 混合专家路由正确性。

    静默 bug 清单：top-k 选择维度错误（选错专家）、缺少二次归一化（
    路由权重之和不等于 1）、死专家（某些专家从未被选中）、
    专家输出加权累加逻辑维度错位。
    """

    @pytest.fixture
    def moe(self, moe_ffn_config):
        """创建 4 专家 top_k=2 的 MoE 模块"""
        from model.ffn import TarsLMMoE
        return TarsLMMoE(moe_ffn_config)

    def test_output_shape(self, moe):
        """MoE 输出形状与输入一致"""
        out = moe(torch.randn(2, 8, 64))
        assert out.shape == (2, 8, 64)

    def test_each_token_activates_exactly_topk(self, moe):
        """每个 token 恰好路由到 top_k 个专家，且不同 token 可路由到不同专家。

        固定 router 权重使两个 token 分别路由到不同的专家组合，再验证
        模块输出等于各自 top-k 专家输出的加权和（权重为二次归一化后的
        路由概率）。若 topk 维度/顺序错误或加权累加错位，输出会偏离期望值。
        """
        # 固定 router：token 0（仅第 0 维非零）路由到专家 {0,1}，
        # token 1（仅第 1 维非零）路由到专家 {2,3}
        with torch.no_grad():
            moe.router.weight.zero_()
            moe.router.weight[0, 0] = 3.0
            moe.router.weight[1, 0] = 2.0
            moe.router.weight[2, 0] = 1.0
            # 专家 3 对第 0 维的权重保持 0
            moe.router.weight[0, 1] = 1.0
            moe.router.weight[2, 1] = 3.0
            moe.router.weight[3, 1] = 2.0
        x = torch.zeros(1, 2, 64)
        x[0, 0, 0] = 1.0  # token 0 → logits [3, 2, 1, 0]
        x[0, 1, 1] = 1.0  # token 1 → logits [1, 0, 3, 2]
        out = moe(x)

        # 手工计算期望：softmax(logits) 取 top-k 后二次归一化，再按专家加权累加
        expected = torch.zeros_like(x)
        for token_index in range(x.shape[1]):
            logits = moe.router.weight @ x[0, token_index]  # (num_experts,)
            probs = F.softmax(logits, dim=-1)
            weights, expert_indices = torch.topk(probs, moe.top_k, dim=-1)
            weights = weights / weights.sum()
            for k in range(moe.top_k):
                expected[0, token_index] += (
                    weights[k]
                    * moe.experts[expert_indices[k]](x[:, token_index:token_index + 1])[0, 0]
                )
        assert_tensors_allclose(out, expected, rtol=1e-4,
                                msg="MoE 按 token 路由到 top-k 专家的输出与期望不一致")

    def test_routing_weights_sum_to_one(self, moe):
        """路由权重二次归一化后，模块输出与手工归一化的期望一致。

        固定 router 使 softmax 后 top-2 概率之和明显小于 1（约 0.88），
        若模块省略二次归一化，输出会等于"未归一化"的期望并偏离真实期望。
        """
        # 固定 router：logits = x[:, 0] * [3, 2, 1, 0]，所有 token 路由到专家 {0,1}
        with torch.no_grad():
            moe.router.weight.zero_()
            moe.router.weight[0, 0] = 3.0
            moe.router.weight[1, 0] = 2.0
            moe.router.weight[2, 0] = 1.0
        x = torch.ones(1, 2, 64)
        out = moe(x)

        # 期望：softmax([3,2,1,0]) 的 top-2 概率，分别做与不做二次归一化
        probs = F.softmax(torch.tensor([3.0, 2.0, 1.0, 0.0]), dim=-1)
        weights, expert_indices = torch.topk(probs, moe.top_k, dim=-1)
        weights_normalized = weights / weights.sum()
        expected_normalized = torch.zeros_like(x)
        expected_raw = torch.zeros_like(x)
        for token_index in range(x.shape[1]):
            for k in range(moe.top_k):
                expert_out = moe.experts[expert_indices[k]](x[:, token_index:token_index + 1])[0, 0]
                expected_normalized[0, token_index] += weights_normalized[k] * expert_out
                expected_raw[0, token_index] += weights[k] * expert_out

        # 模块输出必须等于"二次归一化后"的期望（缺少归一化会整体缩放输出）
        assert_tensors_allclose(out, expected_normalized, rtol=1e-4,
                                msg="MoE 路由权重缺少二次归一化")
        # 同时证明该断言确有区分度：未归一化的期望与模块输出明显不同
        assert not torch.allclose(out, expected_raw, rtol=1e-4, atol=1e-6), (
            "MoE 路由权重未做二次归一化"
        )

    def test_all_experts_reachable(self, moe):
        """所有专家都必须可被路由选中（无死专家）。

        逐个把 router 强指向某个专家（logits 其一远大于其余），
        验证模块输出等于该专家独立处理的结果。若路由无法选中某专家
        或选中后权重/累加异常，对应迭代的断言会失败。
        """
        x = torch.ones(1, 4, 64)
        for expert_index in range(moe.num_experts):
            with torch.no_grad():
                moe.router.weight.zero_()
                moe.router.weight[expert_index, 0] = 100.0
            out = moe(x)
            expected = moe.experts[expert_index](x)
            assert_tensors_allclose(out, expected, rtol=1e-4,
                                    msg=f"专家 {expert_index} 无法被路由选中或路由权重异常")

    def test_topk_1_exact(self):
        """top_k=1 时 MoE 输出应等于唯一选中专家的独立输出。

        静默 bug：加权累加循环中权重计数与 top_k 不匹配，
        导致 top_k=1 时输出不等于单个专家结果。
        """
        from model.ffn import TarsLMMoE
        cfg = TarsLMConfig(hidden_size=64, intermediate_size=256,
                           hidden_dropout=0.0, num_experts=3, moe_top_k=1)
        moe1 = TarsLMMoE(cfg)
        x = torch.randn(1, 1, 64)
        x_flat = x.view(-1, 64)
        router_logits = moe1.router(x_flat)
        _, selected = torch.topk(F.softmax(router_logits, dim=-1), 1, dim=-1)
        expert_idx = selected[0, 0].item()
        expert_out = moe1.experts[expert_idx](x)
        moe_out = moe1(x)
        assert_tensors_allclose(moe_out, expert_out, rtol=1e-4,
                                msg="top_k=1 MoE output should equal the selected expert's output")

    def test_gradient_flow_all_experts(self, moe):
        """路由器和所有专家参数都能收到梯度。

        静默 bug：路由逻辑中的形状错误可能导致某些专家的参数
        与计算图断开，梯度为 None 且静默不报错。
        使用足够大的 batch 使所有专家都至少被选中一次
        （种子固定为 42，结果可复现）。
        """
        moe.train()
        x = torch.randn(64, 8, 64, requires_grad=True)  # 512 个 token
        moe(x).pow(2).mean().backward()
        assert moe.router.weight.grad is not None, "路由器缺少梯度"
        for i, expert in enumerate(moe.experts):
            for name, p in expert.named_parameters():
                assert p.grad is not None, f"专家 {i} 参数 {name} 缺少梯度"

    def test_load_balance_loss_is_finite(self, moe):
        """MoE 前向应产生非负且有限的负载均衡辅助损失。"""
        moe(torch.randn(32, 8, 64))
        assert moe.aux_loss is not None
        load_balance_loss, router_z_loss = moe.aux_loss
        assert load_balance_loss.item() >= 0.0
        assert router_z_loss.item() >= 0.0
        assert torch.isfinite(load_balance_loss)
        assert torch.isfinite(router_z_loss)

    def test_batch_size_1(self, moe):
        """batch=1 边界测试"""
        out = moe(torch.randn(1, 4, 64))
        assert out.shape == (1, 4, 64)

    def test_seq_len_1(self, moe):
        """seq_len=1 边界测试"""
        out = moe(torch.randn(2, 1, 64))
        assert out.shape == (2, 1, 64)
