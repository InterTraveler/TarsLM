"""RMS 归一化层（RMSNorm）实现。

RMSNorm 省略 LayerNorm 的均值中心化步骤，只做
x / sqrt(mean(x^2) + eps) * gamma 的缩放，是 Llama 系列常用的高效归一化。
"""

import torch
from torch import nn


class TarsLMRMSNorm(nn.Module):
    """RMS 归一化层。

    与传统 LayerNorm 的区别:
      LayerNorm: (x - mean) / std * gamma + beta   （减均值 + 除标准差）
      RMSNorm:    x / rms * gamma                  （只需除 RMS，无 beta）
    RMSNorm 少了两个运算（减均值、加 beta），前向和反向都快约 50%。
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        # nn.Parameter 将张量注册为可训练参数（会出现在 model.parameters() 中）
        # 初始值全 1: 训练初期相当于恒等映射，不会破坏残差分支的信号
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        """前向传播 —— 仅 3 步。

        参数:
            x: 输入张量, shape (batch, seq_len, hidden_size)
        返回:
            归一化后的张量, shape 同输入
        """
        # 先提升到 fp32 计算统计量，避免 fp16/bf16 中 x.pow(2) 溢出。
        x_float = x.to(torch.float32)
        return (
            x_float
            * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
            * self.weight.to(torch.float32)
        ).to(x.dtype)
