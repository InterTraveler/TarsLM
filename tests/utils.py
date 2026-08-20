"""测试参考实现：用独立公式与自研模块做数值对齐。"""

import torch
import torch.nn.functional as F
from torch import nn


# ===========================================================================
#  _rotate_half 参考实现
# ===========================================================================
def ref_rotate_half(x, cos, sin):
    """参考实现：用复数旋转独立计算 RoPE 变换。
    公式：x' = cos(theta) * x - sin(theta) * rotate(x)
    其中 rotate(x) = [-x2, x1]（将后半取反拼到前半，前半原样拼到后半）
    """
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    # 复数旋转：(x1 + i*x2) * (cos + i*sin) 的实部和虚部
    real = x1 * cos - x2 * sin
    imag = x2 * cos + x1 * sin
    return torch.cat([real, imag], dim=-1)


# ===========================================================================
#  RMSNorm 参考实现
# ===========================================================================
def ref_rms_norm(x, weight, eps=1e-6):
    """参考实现：x / sqrt(mean(x^2) + eps) * weight"""
    rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + eps)
    return x / rms * weight


# ===========================================================================
#  RoPE 参考实现
# ===========================================================================
class RefRotaryEmbedding(nn.Module):
    """RoPE 编码器参考实现。独立计算 cos/sin 表。"""

    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(t, freqs)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, xq, xk):
        s = xq.shape[-2]
        cos = self.cos[:s].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:s].unsqueeze(0).unsqueeze(0)
        return ref_rotate_half(xq, cos, sin), ref_rotate_half(xk, cos, sin)


# ===========================================================================
#  因果注意力掩码工具
# ===========================================================================
def _make_causal_mask(seq_len, device, dtype):
    """生成标准下三角因果掩码：token i 只能关注位置 <= i 的 token。"""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    # True 表示需要屏蔽的位置
    return mask


# ===========================================================================
#  Multi-Head Attention 参考实现（不依赖 SDPA，手写 softmax）
# ===========================================================================
def ref_multi_head_attention(q, k, v, attention_mask=None, is_causal=True, dropout_p=0.0):
    """参考实现：手写 scaled dot-product attention，不使用 SDPA 融合算子。
    用于验证自研 SDPA 调用的正确性。

    输入形状：
        q, k, v: (batch, num_heads, seq_len, head_dim)
    返回：
        attn_output: (batch, num_heads, seq_len, head_dim)
    """
    head_dim = q.shape[-1]
    # 缩放因子：QK^T / sqrt(d)
    scale = head_dim ** 0.5
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

    # 构造注意力掩码
    seq_len = q.shape[-2]
    if is_causal:
        causal_mask = _make_causal_mask(seq_len, q.device, q.dtype)
        # 把因果掩码转为 -inf 加到 attention weights 上
        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    if attention_mask is not None:
        if attention_mask.dim() == 4:
            # (batch, 1, 1, seq) -> 扩展为 (batch, 1, seq, seq)
            mask_4d = attention_mask.to(dtype=q.dtype)
            mask_4d = (1.0 - mask_4d) * float("-inf")
            attn_weights = attn_weights + mask_4d
        elif attention_mask.dim() == 2:
            # (batch, seq) -> (batch, 1, 1, seq)
            mask_2d = attention_mask[:, None, None, :].to(dtype=q.dtype)
            mask_2d = (1.0 - mask_2d) * float("-inf")
            attn_weights = attn_weights + mask_2d

    # 对注意力权重做 softmax 归一化
    attn_weights = F.softmax(attn_weights, dim=-1)

    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # 加权求和
    attn_output = torch.matmul(attn_weights, v)
    return attn_output


# ===========================================================================
#  SwiGLU FFN 参考实现
# ===========================================================================
def ref_swiglu_ffn(x, gate_weight, up_weight, down_weight):
    """参考实现：down(SiLU(x @ gate_weight^T) * (x @ up_weight^T))"""
    gate = F.linear(x, gate_weight)
    up = F.linear(x, up_weight)
    activated = F.silu(gate) * up
    return F.linear(activated, down_weight)


# ===========================================================================
#  DecoderLayer 参考实现
# ===========================================================================
class RefDecoderLayer(nn.Module):
    """单层 Decoder 参考实现：独立组装各组件，用于端到端对齐。"""

    def __init__(self, config):
        super().__init__()
        from model.attention import TarsLMAttention
        from model.ffn import TarsLMFeedForward
        from model.norm import TarsLMRMSNorm
        self.input_layernorm = TarsLMRMSNorm(config.hidden_size, config.norm_eps)
        self.self_attn = TarsLMAttention(config)
        self.post_attention_layernorm = TarsLMRMSNorm(config.hidden_size, config.norm_eps)
        self.mlp = TarsLMFeedForward(config)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self._gradient_checkpointing = False

    def _run(self, fn, *args, **kwargs):
        if self._gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False, **kwargs)
        return fn(*args, **kwargs)

    def forward(self, hidden_states, attention_mask=None, past_key_value=None, use_cache=False):
        # 子层 1：注意力
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, kv_cache = self._run(
            self.self_attn, hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + self.dropout(attn_out)

        # 子层 2：前馈网络
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.dropout(self._run(self.mlp, hidden_states))

        return hidden_states, kv_cache


# ===========================================================================
#  数值一致性断言工具
# ===========================================================================
def assert_tensors_allclose(actual, expected, rtol=1e-4, atol=1e-6, msg=""):
    """统一断言：使用 torch.allclose 而非 ==。"""
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"{msg} max diff: {(actual - expected).abs().max().item():.6e}, "
        f"rtol={rtol}, atol={atol}"
    )


# ===========================================================================
#  参数计数工具
# ===========================================================================
def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


def count_trainable_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
