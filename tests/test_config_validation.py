"""TarsLMConfig 架构参数校验测试。"""

import pytest

from model.config import TarsLMConfig


def test_valid_config_creates():
    config = TarsLMConfig(
        vocab_size=128,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=2,
        moe_top_k=1,
    )
    assert config.hidden_size == 64


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hidden_size": 63, "num_attention_heads": 4}, "hidden_size"),
        ({"hidden_size": 72, "num_attention_heads": 6, "num_key_value_heads": 4}, "num_key_value_heads"),
        ({"num_experts": 2, "moe_top_k": 3}, "moe_top_k"),
        ({"pad_token_id": 999, "vocab_size": 128}, "pad_token_id"),
        ({"rope_theta": 0}, "rope_theta"),
    ],
)
def test_invalid_config_raises(kwargs, message):
    base = {
        "vocab_size": 128,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        TarsLMConfig(**base)
