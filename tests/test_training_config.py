"""训练配置辅助函数的单元测试。

本文件只验证配置字典的合并、路径解析以及 DeepSpeed 开关逻辑，不会创建
任何 TarsLMModel。服务器配置的模型参数量约为 6.7B，普通开发环境无法承受，
因此这些测试刻意避开模型实例化和数据加载，只做轻量级配置检查。
"""

import json
from pathlib import Path

import pytest
import yaml

from common.training_utils import (
    apply_finetune_overrides,
    merge_pretrain_sections,
    resolve_config_paths,
    resolve_deepspeed_config_path,
    validate_training_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"


def load_config(name: str) -> dict:
    """使用 UTF-8 读取项目 ``config`` 目录下的 YAML 配置。"""
    with open(CONFIG_DIR / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_default_and_consumer_keep_plain_ddp():
    """未配置 DeepSpeed 的默认配置必须继续使用普通 DDP。"""
    for name in ("default.yaml", "consumer.yaml"):
        cfg = load_config(name)
        # 返回 None 表示入口脚本不会给 TrainingArguments 传入 DeepSpeed 配置。
        assert resolve_deepspeed_config_path(cfg, PROJECT_ROOT) is None


def test_server_enables_existing_zero2_config():
    """server.yaml 必须指向真实存在的 ZeRO-2 配置文件。"""
    cfg = load_config("server.yaml")
    path = resolve_deepspeed_config_path(cfg, PROJECT_ROOT)

    assert path is not None
    assert Path(path).is_file()

    # 加载并检查 JSON，确保不是只配置了路径但文件内容错误。
    ds_cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    assert ds_cfg["zero_optimization"]["stage"] == 2
    assert ds_cfg["bf16"]["enabled"] == "auto"


def test_merge_pretrain_sections():
    """pretrain 子段应被提升到顶层，并且不修改原始配置字典。"""
    cfg = {
        "model": {"hidden_size": 16},
        "pretrain": {
            "training": {"max_steps": 10},
            "dataset": {"data_format": "json"},
            "paths": {"checkpoint_dir": "./checkpoints/"},
        },
    }

    merged = merge_pretrain_sections(cfg)

    # 顶层字段可被入口脚本直接读取。
    assert merged["training"] == {"max_steps": 10}
    assert merged["dataset"]["data_format"] == "json"
    assert merged["paths"]["checkpoint_dir"] == "./checkpoints/"
    # 输入字典不应因为合并操作而被污染。
    assert "training" not in cfg


def test_finetune_overrides_merge_and_preserve_defaults():
    """finetune 只覆盖指定字段，其余字段继续继承 pretrain 默认值。"""
    cfg = {
        "pretrain": {
            "training": {
                "max_steps": 50000,
                "warmup_steps": 2000,
                "learning_rate": 3.0e-4,
            },
            "dataset": {
                "data_format": "jsonl",
                "train_data_path": "/data/pretrain/",
            },
            "paths": {"checkpoint_dir": "./checkpoints/"},
        },
        "finetune": {
            "training": {"max_steps": 8000, "warmup_steps": 100},
            "dataset": {"train_data_path": "/data/chat/"},
            "paths": {"checkpoint_dir": "./checkpoints_chat_server/"},
        },
    }

    merged = apply_finetune_overrides(cfg)

    # 被覆盖的字段应使用 finetune 的值。
    assert merged["training"]["max_steps"] == 8000
    assert merged["training"]["warmup_steps"] == 100
    # 未覆盖的字段应保留 pretrain 的值。
    assert merged["training"]["learning_rate"] == 3.0e-4
    assert merged["dataset"]["train_data_path"] == "/data/chat/"
    assert merged["dataset"]["data_format"] == "jsonl"
    assert merged["paths"]["checkpoint_dir"] == "./checkpoints_chat_server/"


def test_server_token_budget_matches_comment():
    """server.yaml 注释中的 token 总量必须与实际计算值一致。"""
    cfg = load_config("server.yaml")
    training = cfg["pretrain"]["training"]

    # 1 GPU micro-batch × 32 梯度累积 × 8 卡 × 4096 序列 × 50000 步。
    tokens = (
        training["batch_size"]
        * training["gradient_accumulation_steps"]
        * 8
        * cfg["model"]["max_seq_len"]
        * training["max_steps"]
    )

    assert tokens == 52_428_800_000


def test_server_finetune_warmup_is_explicit():
    """服务器微调不应继承预训练过长的 warmup_steps。"""
    cfg = load_config("server.yaml")

    assert cfg["finetune"]["training"]["warmup_steps"] == 100

    # 合并后的最终配置也必须保持微调自己的 warmup 值。
    merged = apply_finetune_overrides(cfg)
    assert merged["training"]["warmup_steps"] == 100
    assert merged["training"]["max_steps"] == 8000
    assert merged["training"]["gradient_accumulation_steps"] == 1


def test_resolve_config_paths_uses_project_root(tmp_path):
    """相对路径应解析到项目根目录，而不是当前工作目录。"""
    cfg = {
        "pretrain": {
            "dataset": {
                "train_data_path": "./data",
                "tokenizer_path": "./tokenizer",
            },
            "paths": {"checkpoint_dir": "./checkpoints"},
        },
        "finetune": {"pretrained_checkpoint": "./pretrained"},
    }
    merged = merge_pretrain_sections(cfg)
    resolved = resolve_config_paths(merged, tmp_path)

    assert resolved["dataset"]["train_data_path"] == str(
        (tmp_path / "data").resolve()
    )
    assert resolved["dataset"]["tokenizer_path"] == str(
        (tmp_path / "tokenizer").resolve()
    )
    assert resolved["paths"]["checkpoint_dir"] == str(
        (tmp_path / "checkpoints").resolve()
    )
    assert resolved["finetune"]["pretrained_checkpoint"] == str(
        (tmp_path / "pretrained").resolve()
    )


def test_validate_training_config_rejects_missing_sections():
    with pytest.raises(ValueError, match="training"):
        validate_training_config({"model": {}, "moe": {}, "hardware": {}})
