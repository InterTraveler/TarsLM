"""Checkpoint 目录发现工具。"""

import re
from pathlib import Path


def find_latest_checkpoint(checkpoints_root: str, prefix: str = "") -> str:
    """返回步数最大的 ``checkpoint-n`` 目录。

    传入具体 ``checkpoint-n`` 目录时直接返回；传入根目录时扫描子目录。
    ``prefix`` 仅用于日志前缀，让预训练、微调和推理入口保持各自标识。
    """
    root = Path(checkpoints_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoints 目录不存在: {root}")

    direct_match = re.fullmatch(r"checkpoint-(\d+)", root.name)
    if direct_match:
        print(f"{prefix} 加载指定 checkpoint: checkpoint-{direct_match.group(1)}")
        return str(root)

    best_step = -1
    best_dir = None
    for child in root.iterdir():
        match = re.fullmatch(r"checkpoint-(\d+)", child.name)
        if not match or not child.is_dir():
            continue
        step = int(match.group(1))
        if step > best_step:
            best_step = step
            best_dir = child

    if best_dir is None:
        raise FileNotFoundError(f"{root} 下未找到任何 checkpoint-n 目录")
    print(f"{prefix} 自动选择最新 checkpoint: checkpoint-{best_step}")
    return str(best_dir)
