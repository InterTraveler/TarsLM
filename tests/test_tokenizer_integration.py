"""真实分词器与模型配置的集成测试。"""

from pathlib import Path

import pytest
from transformers import AutoTokenizer

from chat_finetune import (
    _validate_chat_control_tokens,
    _validate_tokenizer_vocab_size,
)
from main_pretrain import build_model_config
from train_tokenizer import CHAT_CONTROL_TOKENS, train_tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_model_config_reads_real_tokenizer_ids(tmp_path):
    """模型配置必须读取 HF tokenizer 实际分配的 special token ID。"""
    corpus = PROJECT_ROOT / "data" / "train_data" / "multi"
    train_tokenizer(
        corpus_path=str(corpus),
        output_dir=str(tmp_path),
        vocab_size=8000,
        model_type="unigram",
        data_format="json",
        no_progress=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(tmp_path))
    config = build_model_config(
        {
            "model": {
                "hidden_size": 64,
                "num_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "intermediate_size": 128,
                "max_seq_len": 64,
            },
            "moe": {"enabled": False},
            "hardware": {},
        },
        tokenizer,
    )

    assert config.vocab_size == len(tokenizer)
    assert config.pad_token_id == tokenizer.pad_token_id
    assert config.bos_token_id == tokenizer.bos_token_id
    assert config.eos_token_id == tokenizer.eos_token_id
    assert 0 <= config.pad_token_id < config.vocab_size
    assert 0 <= config.bos_token_id < config.vocab_size
    assert 0 <= config.eos_token_id < config.vocab_size


@pytest.mark.parametrize("model_type", ["unigram", "bpe"])
def test_train_tokenizer_reserves_chat_control_tokens(tmp_path, model_type):
    """训练阶段必须把聊天控制 token 预留在普通子词区间之外。"""
    corpus = PROJECT_ROOT / "data" / "train_data" / "multi"
    train_tokenizer(
        corpus_path=str(corpus),
        output_dir=str(tmp_path),
        vocab_size=8000,
        model_type=model_type,
        data_format="json",
        no_progress=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(tmp_path))
    added_vocab = tokenizer.get_added_vocab()
    normal_vocab_size = len(tokenizer) - len(added_vocab)

    for token in CHAT_CONTROL_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        assert token in added_vocab
        assert token_id != tokenizer.unk_token_id
        assert token_id >= normal_vocab_size
    _validate_chat_control_tokens(tokenizer)


def test_validate_rejects_normal_subword_chat_control_token():
    """旧分词器把控制 token 学成普通子词时必须被拒绝。"""

    class _LegacyTokenizer:
        unk_token_id = 0

        def __len__(self):
            return 6

        def get_added_vocab(self):
            return {
                "<unk>": 0,
                "<pad>": 3,
                "<s>": 4,
                "</s>": 5,
            }

        def convert_tokens_to_ids(self, token):
            return {
                "<|user|>": 0,
                "<|assistant|>": 1,
                "<|end|>": 0,
            }[token]

    with pytest.raises(ValueError, match="未落在预留 ID 区间"):
        _validate_chat_control_tokens(_LegacyTokenizer())


def test_unigram_vocab_size_matches_requested(tmp_path):
    """Unigram 训练后的词表长度必须等于配置的 vocab_size。"""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "data.txt").write_text(
        "\n".join(f"tok{index}" for index in range(3000)),
        encoding="utf-8",
    )

    train_tokenizer(
        corpus_path=str(corpus),
        output_dir=str(tmp_path / "out"),
        vocab_size=400,
        model_type="unigram",
        data_format="txt",
        no_progress=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(tmp_path / "out"))
    assert len(tokenizer) == 400


def test_chat_vocab_size_rejects_smaller_tokenizer():
    """微调分词器词表小于 checkpoint 时必须显式报错。"""
    with pytest.raises(ValueError, match="小于 checkpoint 模型词表大小"):
        _validate_tokenizer_vocab_size(
            tokenizer_vocab_size=100,
            model_vocab_size=101,
        )

    _validate_tokenizer_vocab_size(
        tokenizer_vocab_size=100,
        model_vocab_size=100,
    )
    _validate_tokenizer_vocab_size(
        tokenizer_vocab_size=101,
        model_vocab_size=100,
    )
