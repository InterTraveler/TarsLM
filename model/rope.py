"""RoPE 旋转位置编码实现。

包含核心旋转变换 _rotate_half 与预计算 cos/sin 频率表的
TarsLMRotaryEmbedding，用于在 Attention 前对 query/key 施加相对位置信息。
"""

import torch
from torch import nn


def _rotate_half(x, cos, sin):
    """RoPE 核心旋转变换 —— 复数旋转公式的实部/虚部分解。

    将输入 x 沿最后一维对半拆分为 (x1, x2)，按 e^(i*theta) 复数旋转展开：
        real = x1 * cos - x2 * sin    （实部）
        imag = x2 * cos + x1 * sin    （虚部）

    参数:
        x:   待旋转的张量, shape (..., head_dim)
        cos: 预计算的 cos(theta), shape (..., head_dim//2)
        sin: 预计算的 sin(theta), shape (..., head_dim//2）
    返回:
        旋转后的张量, shape 同 x
    """
    d2 = x.shape[-1] // 2                # 对半拆分点: head_dim / 2
    x1, x2 = x[..., :d2], x[..., d2:]    # 将最后一维拆成前后两半
    return torch.cat([
        x1 * cos - x2 * sin,             # 实部: 前半按旋转矩阵变换
        x2 * cos + x1 * sin,             # 虚部: 后半按旋转矩阵变换
    ], dim=-1)                            # 沿最后一维拼接回去


class TarsLMRotaryEmbedding(nn.Module):
    """旋转位置编码器：在 __init__ 中预计算 cos/sin 频率表, forward 时直接切片查表。

    频率公式: theta_i = base^(-2i/dim), i = 0, 1, ..., dim/2-1
    base 越大越能支持长序列外推（Llama 默认 10000, Llama 3 用 500000）

    预计算的 cos/sin 注册为 buffer:
    - persistent=False: 不保存到 state_dict（forward 时可重新计算，省磁盘）
    - requires_grad=False: buffer 默认不参与梯度更新
    """

    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()

        # ---- 第 1 步: 计算每个特征维度对的旋转频率 theta_i ----
        # torch.arange(0, dim, 2) -> [0, 2, 4, ..., dim-2]
        # base^(-2i/dim): 频率沿维度指数衰减，低频（i 小）对应长距离依赖
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

        # ---- 第 2 步: 为每个位置计算角度 pos * theta_i ----
        t = torch.arange(max_seq_len, dtype=torch.float32)  # 位置索引 [0, 1, 2, ...]
        angles = torch.outer(t, freqs)                       # 外积: (max_seq_len, dim//2)

        # ---- 第 3 步: 取 cos/sin 并注册为 buffer ----
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, xq, xk, position_ids=None):
        """对 query 和 key 施加旋转位置编码。

        注意: 不对 value 施加 RoPE！RoPE 的设计原则是位置信息只影响
        "谁关注谁"（Attention Score），不影响"关注的内容"（Attention Output）。

       参数:
            xq: query 张量, shape (batch, num_heads, seq_len, head_dim)
            xk: key   张量, shape (batch, num_heads, seq_len, head_dim)
            position_ids: 可选绝对位置索引，shape (batch, seq_len)
        返回:
            (旋转后的 query, 旋转后的 key)
        """
        batch_size, _, seq_len, _ = xq.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=xq.device, dtype=torch.long)
        else:
            position_ids = position_ids.to(xq.device, dtype=torch.long)

        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        elif position_ids.shape[0] == 1 and batch_size != 1:
            position_ids = position_ids.expand(batch_size, -1)

        # 位置索引在 batch 维展开，head 维插入 1 以适配 Q/K 的四维形状。
        # cos/sin 以 float32 buffer 保存，但必须跟随输入 dtype，否则 fp16/bf16
        # 训练时 q/k 会被提升为 float32，破坏混合精度并增加显存占用。
        cos = self.cos[position_ids].unsqueeze(1).to(dtype=xq.dtype)
        sin = self.sin[position_ids].unsqueeze(1).to(dtype=xq.dtype)
        return _rotate_half(xq, cos, sin), _rotate_half(xk, cos, sin)
