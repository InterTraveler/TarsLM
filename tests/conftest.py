"""pytest 全局 fixtures：固定随机种子并配置离线 HF 测试环境。"""

import os
import random
import tempfile

import numpy as np
import pytest
import torch

from model.config import TarsLMConfig

# 固定 HF 缓存与离线模式，避免本地数据测试受默认缓存目录/网络影响
os.environ["HF_HOME"] = os.path.join(tempfile.gettempdir(), "tarslm_pytest_hf")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


def _set_all_seeds(seed=42):
    """固定所有随机种子, 保证测试可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.fixture(scope="session", autouse=True)
def global_seed():
    """会话级: 整个测试进程设置一次种子"""
    _set_all_seeds(42)


@pytest.fixture(autouse=True)
def per_test_seed():
    """函数级: 每个测试用例运行前重置种子"""
    _set_all_seeds(42)


@pytest.fixture
def attn_config():
    """Attention 模块测试配置"""
    return TarsLMConfig(hidden_size=64, num_attention_heads=4,
                        num_key_value_heads=4, max_seq_len=128, attention_dropout=0.0,
                        hidden_dropout=0.0, rope_theta=10000.0, use_bias=False)


@pytest.fixture
def ffn_config():
    """FFN 模块测试配置"""
    return TarsLMConfig(hidden_size=64, intermediate_size=256,
                        hidden_dropout=0.0, use_bias=False)


@pytest.fixture
def layer_config():
    """DecoderLayer 测试配置"""
    return TarsLMConfig(hidden_size=64, num_attention_heads=4,
                        num_key_value_heads=4, intermediate_size=256, max_seq_len=128,
                        num_layers=1, norm_eps=1e-6, attention_dropout=0.0, hidden_dropout=0.0,
                        use_bias=False, use_moe=False, rope_theta=10000.0)


@pytest.fixture
def model_config():
    """完整模型端到端测试配置"""
    return TarsLMConfig(vocab_size=128, hidden_size=64, num_layers=2,
                        num_attention_heads=4, num_key_value_heads=4, intermediate_size=256,
                        max_seq_len=64, norm_eps=1e-6, attention_dropout=0.0, hidden_dropout=0.0,
                        use_bias=False, use_moe=False,
                        rope_theta=10000.0, initializer_range=0.02,
                        tie_word_embeddings=False, pad_token_id=0, bos_token_id=1, eos_token_id=2)


@pytest.fixture
def gqa_config():
    """GQA: num_heads=8, num_kv_heads=2, 4 Q heads per KV pair."""
    return TarsLMConfig(hidden_size=64, num_attention_heads=8,
                        num_key_value_heads=2, max_seq_len=128, attention_dropout=0.0,
                        hidden_dropout=0.0, rope_theta=10000.0, use_bias=False)


@pytest.fixture
def moe_ffn_config():
    """MoE FFN: 4 experts, top_k=2."""
    return TarsLMConfig(hidden_size=64, intermediate_size=256,
                        hidden_dropout=0.0, use_bias=False,
                        use_moe=True, num_experts=4, moe_top_k=2)


@pytest.fixture
def moe_layer_config():
    """DecoderLayer with MoE enabled."""
    return TarsLMConfig(hidden_size=64, num_attention_heads=4,
                        num_key_value_heads=4, intermediate_size=256, max_seq_len=128,
                        num_layers=1, norm_eps=1e-6, attention_dropout=0.0, hidden_dropout=0.0,
                        use_bias=False, use_moe=True, num_experts=2, moe_top_k=1,
                        rope_theta=10000.0)
