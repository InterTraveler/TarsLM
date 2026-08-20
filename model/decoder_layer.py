"""TarsLM Decoder 单层。

每个 DecoderLayer 由 Self-Attention 与 FFN/MoE 两个子层组成，
均采用 Pre-Norm + 残差结构，并按 config.use_moe 切换前馈实现。
"""

import torch
from torch import nn

from model.attention import TarsLMAttention
from model.ffn import TarsLMFeedForward, TarsLMMoE
from model.norm import TarsLMRMSNorm


class TarsLMDecoderLayer(nn.Module):
    """单层 Decoder: Attention + FFN/MoE，均采用 Pre-Norm + 残差结构。

    内部结构:
      input_layernorm          —— Attention 前的 Pre-Norm
      self_attn               —— 多头自注意力（支持 MHA/GQA/Flash Attn）
      post_attention_layernorm —— FFN/MoE 前的 Pre-Norm
      mlp                     —— 前馈网络（SwiGLU 或 MoE，按 config.use_moe 切换）
      dropout                 —— 残差 dropout（训练时随机丢弃，eval 时自动关闭）
    """

    def __init__(self, config):
        super().__init__()

        # ---- 梯度检查点开关（默认关闭，由 Trainer 在 training 时通过 model 层开启） ----
        self._gradient_checkpointing = False

        # ---- 第一个 Pre-Norm: Attention 之前做归一化 ----
        self.input_layernorm = TarsLMRMSNorm(config.hidden_size, config.norm_eps)

        # ---- 多头自注意力 ----
        self.self_attn = TarsLMAttention(config)

        # ---- 第二个 Pre-Norm: FFN 之前做归一化 ----
        # 注意: 两个 Norm 层是独立的 Parameter 对象，权重不共享
        self.post_attention_layernorm = TarsLMRMSNorm(config.hidden_size, config.norm_eps)

        # ---- 前馈网络: 按配置切换 MoE / 标准 FFN ----
        if config.use_moe:
            self.mlp = TarsLMMoE(config)  # 混合专家（稀疏激活）
        else:
            self.mlp = TarsLMFeedForward(config)  # 标准 SwiGLU（密集计算）

        # ---- 残差 Dropout ----
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states, attention_mask=None, past_key_value=None, use_cache=False,
                position_ids=None):
        """前向传播 —— Pre-Norm + Attention → 残差 → Pre-Norm + FFN → 残差。

        参数:
            hidden_states:  (batch, seq_len, hidden_size)
            attention_mask: 注意力掩码
            past_key_value: KV 缓存
            use_cache:      是否返回 KV 缓存
        返回:
            (hidden_states, kv_cache)
        """
        # ================================================================
        # 子层 1: Self-Attention（Pre-Norm + 残差连接）
        # ================================================================
        residual = hidden_states  # 保存残差（输入本身）
        hidden_states = self.input_layernorm(hidden_states)  # Pre-Norm: 先归一化
        attn_out, kv_cache = self._run(  # 自注意力计算
            self.self_attn, hidden_states,  # _run 支持梯度检查点
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            position_ids=position_ids,
        )
        hidden_states = residual + self.dropout(attn_out)  # 残差连接: 输入 + Attn 输出

        # ================================================================
        # 子层 2: FFN / MoE（Pre-Norm + 残差连接）
        # ================================================================
        residual = hidden_states  # 再次保存残差
        hidden_states = self.post_attention_layernorm(hidden_states)  # Pre-Norm: 先归一化
        hidden_states = residual + self.dropout(  # 残差连接: 输入 + FFN 输出
            self._run(self.mlp, hidden_states)  # FFN/MoE 前向
        )

        return hidden_states, kv_cache

    def _run(self, fn, *args, **kwargs):
        """梯度检查点包装器 —— 根据 _gradient_checkpointing 标志决定是否启用。

        正常训练:  PyTorch 保存每层中间激活值 → 显存大
        检查点模式: 前向时不保存 → 反向时重新计算 → 显存小、速度慢
        torch.utils.checkpoint.checkpoint 自动管理"不保存 → 重算"过程

        use_reentrant=False: PyTorch 2.0+ 推荐的非重入模式（避免嵌套 autograd 问题）
        """
        if self._gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False, **kwargs)
        return fn(*args, **kwargs)
