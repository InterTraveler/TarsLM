"""TarsLM 默认配置的一键初始化脚本。

默认按顺序执行三步：
1. 生成默认分词器；
2. 运行默认预训练配置；
3. 在预训练 checkpoint 上运行默认对话微调配置。

适合新 clone 后快速验证完整训练链路。可通过 ``--skip-tokenizer``、
``--skip-pretrain``、``--skip-finetune`` 跳过指定阶段。
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from train_tokenizer import train_tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


def parse_args() -> argparse.Namespace:
    """解析一键初始化命令行参数。"""
    parser = argparse.ArgumentParser(description="一键完成默认分词器、预训练和微调")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="默认配置路径，默认使用 config/default.yaml",
    )
    parser.add_argument(
        "--model_type",
        choices=("unigram", "bpe"),
        default="unigram",
        help="分词算法：unigram 或 bpe",
    )
    parser.add_argument(
        "--skip_tokenizer",
        action="store_true",
        help="跳过分词器训练阶段",
    )
    parser.add_argument(
        "--skip_pretrain",
        action="store_true",
        help="跳过预训练阶段",
    )
    parser.add_argument(
        "--skip_finetune",
        action="store_true",
        help="跳过对话微调阶段",
    )
    parser.add_argument("--no_progress", action="store_true", help="关闭进度条")
    return parser.parse_args()


def _resolve_path(path: Path) -> Path:
    """把配置中的相对路径解析为项目根目录下的绝对路径。"""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def _load_config(config_path: Path) -> dict:
    """读取 YAML 配置。"""
    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _run_tokenizer(args: argparse.Namespace, cfg: dict) -> None:
    """训练默认分词器，每次启动都会根据当前语料重新训练。"""
    pretrain = cfg.get("pretrain", {})
    pretrain_dataset = pretrain.get("dataset", {})
    finetune = cfg.get("finetune", {})
    finetune_dataset = finetune.get("dataset", {})

    pretrain_corpus = _resolve_path(
        Path(pretrain_dataset.get("train_data_path", "./data/train_data/multi/")),
    )
    finetune_corpus_path = finetune_dataset.get("train_data_path")
    corpus_paths = [pretrain_corpus]
    if finetune_corpus_path:
        finetune_corpus = _resolve_path(Path(finetune_corpus_path))
        if finetune_corpus != pretrain_corpus:
            corpus_paths.append(finetune_corpus)

    output = _resolve_path(
        Path(pretrain_dataset.get("tokenizer_path", "./data/tokenizer/")),
    )

    data_format = pretrain_dataset.get("data_format", "json")
    vocab_size = int(cfg.get("model", {}).get("vocab_size", 8000))
    print("[bootstrap] 开始训练分词器...")
    train_tokenizer(
        corpus_path=str(pretrain_corpus),
        output_dir=str(output),
        vocab_size=vocab_size,
        model_type=args.model_type,
        corpus_paths=[str(path) for path in corpus_paths],
        data_format=data_format,
        no_progress=args.no_progress,
    )


def _run_entry(script: str, config_path: Path, stage_name: str) -> None:
    """用当前 Python 解释器运行训练入口脚本。"""
    command = [
        sys.executable,
        str(PROJECT_ROOT / script),
        "--config",
        str(config_path.resolve()),
    ]
    print(f"[bootstrap] 开始{stage_name}...")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    """依次执行分词器、预训练和微调。"""
    args = parse_args()
    cfg = _load_config(args.config)

    if not args.skip_tokenizer:
        _run_tokenizer(args, cfg)
    else:
        print("[bootstrap] 跳过分词器训练")

    if not args.skip_pretrain:
        _run_entry("main_pretrain.py", args.config, "预训练")
    else:
        print("[bootstrap] 跳过预训练")

    if not args.skip_finetune:
        _run_entry("chat_finetune.py", args.config, "对话微调")
    else:
        print("[bootstrap] 跳过对话微调")

    print("[bootstrap] 一键流程执行完成")


if __name__ == "__main__":
    main()
