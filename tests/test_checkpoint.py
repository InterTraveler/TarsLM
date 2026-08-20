"""Checkpoint 目录发现工具的单元测试。"""

import pytest

from common.checkpoint import find_latest_checkpoint


def test_find_latest_checkpoint_returns_direct_checkpoint(tmp_path):
    """传入具体 checkpoint-n 目录时应直接返回该目录。"""
    checkpoint = tmp_path / "checkpoint-12"
    checkpoint.mkdir()
    assert find_latest_checkpoint(str(checkpoint)) == str(checkpoint)


def test_find_latest_checkpoint_scans_numeric_step(tmp_path):
    """根目录模式应选择步数最大的 checkpoint，并忽略非法子目录。"""
    (tmp_path / "checkpoint-3").mkdir()
    (tmp_path / "checkpoint-20").mkdir()
    (tmp_path / "checkpoint-9").mkdir()
    (tmp_path / "checkpoint-bad").mkdir()

    latest = find_latest_checkpoint(str(tmp_path))
    assert latest == str(tmp_path / "checkpoint-20")


def test_find_latest_checkpoint_raises_when_empty(tmp_path):
    """没有 checkpoint-n 子目录时应抛出 FileNotFoundError。"""
    with pytest.raises(FileNotFoundError, match="未找到任何 checkpoint-n"):
        find_latest_checkpoint(str(tmp_path))
