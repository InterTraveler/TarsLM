"""TarsLM 模型包，统一导出配置与各网络模块。"""

from transformers import AutoConfig, AutoModelForCausalLM

from model.attention import TarsLMAttention
from model.config import TarsLMConfig
from model.decoder_layer import TarsLMDecoderLayer
from model.ffn import TarsLMFeedForward, TarsLMMoE
from model.model import TarsLMModel
from model.norm import TarsLMRMSNorm
from model.rope import TarsLMRotaryEmbedding, _rotate_half

# 注册到 Hugging Face Auto 系列，让 AutoConfig.from_pretrained() 和
# AutoModelForCausalLM.from_pretrained() 可以直接识别 "tarslm" 配置。
AutoConfig.register("tarslm", TarsLMConfig)
AutoModelForCausalLM.register(TarsLMConfig, TarsLMModel)

__all__ = [
    "TarsLMAttention",
    "TarsLMConfig",
    "TarsLMDecoderLayer",
    "TarsLMFeedForward",
    "TarsLMMoE",
    "TarsLMModel",
    "TarsLMRMSNorm",
    "TarsLMRotaryEmbedding",
    "_rotate_half",
]
