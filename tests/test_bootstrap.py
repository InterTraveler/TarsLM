"""bootstrap 一键流程的轻量测试。"""

import sys
from argparse import Namespace

import bootstrap


def test_parse_args_defaults_runs_all_stages(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bootstrap.py"])
    args = bootstrap.parse_args()
    assert args.skip_tokenizer is False
    assert args.skip_pretrain is False
    assert args.skip_finetune is False


def test_resolve_path_uses_project_root():
    resolved = bootstrap._resolve_path(bootstrap.Path("./data/tokenizer"))
    assert resolved == bootstrap.PROJECT_ROOT / "data" / "tokenizer"


def test_run_tokenizer_uses_pretrain_and_finetune_data(monkeypatch):
    """分词器训练语料应同时包含预训练目录和微调目录。"""
    captured = {}

    def fake_train_tokenizer(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(bootstrap, "train_tokenizer", fake_train_tokenizer)
    args = Namespace(model_type="unigram", no_progress=True)
    cfg = {
        "model": {"vocab_size": 8000},
        "pretrain": {
            "dataset": {
                "train_data_path": "./data/train_data/multi/",
                "tokenizer_path": "./data/tokenizer/",
                "data_format": "json",
            }
        },
        "finetune": {
            "dataset": {"train_data_path": "./data/train_data/chat/"}
        },
    }

    bootstrap._run_tokenizer(args, cfg)

    assert captured["corpus_paths"] == [
        str(bootstrap._resolve_path(bootstrap.Path("./data/train_data/multi/"))),
        str(bootstrap._resolve_path(bootstrap.Path("./data/train_data/chat/"))),
    ]
    assert captured["data_format"] == "json"
