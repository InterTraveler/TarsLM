"""TarsLM 前馈网络实现。

包含标准 SwiGLU 密集前馈网络 TarsLMFeedForward 与混合专家
TarsLMMoE（稀疏激活，按 top-k 路由选择专家计算）。
"""

import torch
import torch.nn.functional as F
from torch import nn


class TarsLMFeedForward(nn.Module):
    """SwiGLU 前馈网络: 用门控机制控制信息流动。

    SwiGLU 公式: FFN(x) = down( SiLU(gate(x)) * up(x) )
      - gate(x): 门控信号，经过 SiLU 激活后决定"哪些信息可以通过"
      - up(x):   升维后的特征表示（hidden → intermediate）
      - SiLU(gate) * up: 逐元素乘法（Hadamard Product），门控信号 × 特征
      - down(x): 降维回 hidden_size（intermediate → hidden）

    为什么用 SwiGLU 而不是传统 ReLU FFN？
    SwiGLU 的门控机制让网络能自适应地选择信息，在多项基准中优于 ReLU。
    三个线性投影全部无 bias —— LLM 中 bias 不提升效果，反而浪费参数和显存。
    """

    def __init__(self, config):
        super().__init__()

        # ---- 三个线性投影层 ----
        # gate_proj: hidden → intermediate（门控信号，决定哪些信息通过）
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.use_bias)
        # up_proj:   hidden → intermediate（升维投影，把 hidden 映射到更大的空间）
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.use_bias)
        # down_proj: intermediate → hidden（降维投影，把增强后的表示压回 hidden_size）
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.use_bias)

    def forward(self, x):
        """前向传播 —— 仅一行, 但包含了完整的门控+升维+降维流程。

        参数:
            x: 输入, shape (batch, seq, hidden_size)
        返回:
            输出, shape (batch, seq, hidden_size)
        """
        # F.silu = SiLU（Sigmoid Linear Unit）= x * sigmoid(x)，比 ReLU 更平滑
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TarsLMMoE(nn.Module):
    """混合专家（Mixture of Experts）—— 稀疏激活的前馈网络。

    MoE 原理（对标 Mixtral / DeepSeek-V2）:
      1. Router（路由器）: 一个线性层，为每个 token 输出 num_experts 个分数
      2. Top-k 选择: 选分数最高的 top_k 个专家，其余专家不参与计算
      3. Softmax + 归一化: 选中专家的分数转为概率权重，总和归一化到 1
      4. 加权累加: 每个选中的专家独立处理 token，结果按权重累加

    优势: 参数量放大 num_experts 倍，但每 token 只激活 top_k 个专家的计算量。
    例如 num_experts=8, top_k=2: 总参数 8x 但计算量仅 2x。

    静默 bug 防范:
      - 二次归一化: Softmax 后 + topk 后必须再次归一化，确保权重和 = 1
      - 死专家: Router 初始化可能导致某些专家永远不被选中（不参与训练）
      - 维度错位: expert 输出加权累加时维度必须对齐
    """

    def __init__(self, config):
        super().__init__()

        # ---- MoE 核心参数 ----
        self.num_experts = config.num_experts    # 专家总数
        self.top_k = config.moe_top_k            # 每 token 激活的专家数
        self.hidden_size = config.hidden_size
        self.aux_loss = None

        # ---- Router —— 决定每个 token 去哪个专家 ----
        # 输入: (batch*seq, hidden) → 输出: (batch*seq, num_experts)
        # 输出的每个值代表该 token 被路由到对应专家的"倾向性分数"
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=config.use_bias)

        # ---- 专家集合 —— 每个专家是一个完整的 SwiGLU FFN ----
        # ModuleList 自动注册子模块到 parameters() 中
        self.experts = nn.ModuleList([
            TarsLMFeedForward(config) for _ in range(config.num_experts)
        ])

    def forward(self, x):
        """前向传播 —— 稀疏路由 + 专家计算 + 加权累加。

        参数:
            x: 输入, shape (batch, seq, hidden_size)
        返回:
            输出, shape (batch, seq, hidden_size)
        """
        bsz, seq_len, hidden = x.shape

        # ---- 第 1 步: 展平为 (batch*seq, hidden)，每个 token 独立路由 ----
        x_flat = x.view(-1, hidden)                                  # (batch*seq, hidden)

        # ---- 第 2 步: Router 打分 + Softmax + Top-k 选择 ----
        router_logits = self.router(x_flat)                          # (tokens, num_experts)
        routing_probs = F.softmax(router_logits, dim=-1)             # 分数 → 概率分布
        # torch.topk 返回 (values, indices)，这里只需要 indices
        routing_weights, selected_experts = torch.topk(routing_probs, self.top_k, dim=-1)
        # 【关键】二次归一化: topk 后权重之和不等于 1，必须再除一次总和
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # 计算可微的负载均衡损失和 router z-loss，供上层模型附加到语言模型损失。
        self.aux_loss = self._compute_aux_loss(
            routing_probs=routing_probs,
            selected_experts=selected_experts,
            router_logits=router_logits,
        )

        # ---- 第 3 步: 初始化结果张量（全零，后续累加各专家输出） ----
        final_output = torch.zeros_like(x_flat)

        # ---- 第 4 步: 逐专家处理 ----
        for expert_idx in range(self.num_experts):
            # torch.where 找到所有路由到当前专家的 token 索引
            batch_idx, topk_idx = torch.where(selected_experts == expert_idx)
            if batch_idx.numel() == 0:
                continue  # 当前专家无 token 选中，跳过（稀疏激活）

            # 取出该专家负责的 token → 送入专家 FFN
            expert_input = x_flat[batch_idx]                         # (n_tokens, hidden)
            expert_output = self.experts[expert_idx](expert_input.unsqueeze(1)).squeeze(1)  # (n_tokens, hidden)

            # 取出对应的路由权重并加权
            weight = routing_weights[batch_idx, topk_idx].unsqueeze(-1)  # (n_tokens, 1)
            final_output[batch_idx] += weight * expert_output        # 累加到结果中

        # ---- 第 5 步: 恢复原始形状 (batch, seq, hidden) ----
        return final_output.view(bsz, seq_len, hidden)

    def _compute_aux_loss(self, routing_probs, selected_experts, router_logits):
        """返回 (load_balance_loss, router_z_loss)。"""
        num_tokens = routing_probs.shape[0]
        importance = routing_probs.mean(dim=0)
        one_hot = F.one_hot(selected_experts, num_classes=self.num_experts).float()
        load = one_hot.sum(dim=(0, 1)) / max(num_tokens * self.top_k, 1)
        load_balance_loss = (importance * load).sum() * self.num_experts
        router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        return load_balance_loss, router_z_loss
