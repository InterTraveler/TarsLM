"""语料加载模块的单元测试。

覆盖递归文件发现、data_format 过滤、eval_*/valid_* 验证集分流、
多格式混合加载以及 load_train_eval_data 端到端行为。
"""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import IterableDataset

from data.data_loader import discover_data_files, is_eval_file, load_split, load_train_eval_data


def _make_corpus(root: Path) -> Path:
    """构造一个混合格式的临时语料目录，供各测试用例复用。

    目录结构：
      corpus/
        train_data.json              训练用 JSON 数组（2 条）
        train_data.jsonl             训练用 JSONL（2 条）
        notes.txt                    训练用纯文本（2 行）
        sub/part-0000.snappy.parquet 训练用 Snappy Parquet（2 条）
        eval_data.json               验证用 JSON（2 条）
        valid_data.jsonl             验证用 JSONL（2 条）
    """
    # 子目录用于验证递归扫描能够穿透到深层目录
    sub = root / "sub"
    sub.mkdir(parents=True)

    # JSON 数组格式：整个文件是一个 [{"text": ...}, ...] 列表
    (root / "train_data.json").write_text(
        json.dumps([{"text": "json one"}, {"text": "json two"}], ensure_ascii=False),
        encoding="utf-8",
    )

    # JSONL 格式：每行一个 JSON 对象，逐行解析
    (root / "train_data.jsonl").write_text(
        '{"text": "jsonl one"}\n{"text": "jsonl two"}\n',
        encoding="utf-8",
    )

    # 纯文本格式：每行直接作为一条语料
    (root / "notes.txt").write_text("txt one\ntxt two\n", encoding="utf-8")

    # Parquet 格式：使用 .snappy.parquet 后缀，验证 HF 可识别该扩展名
    pq.write_table(
        pa.table({"text": ["parquet one", "parquet two"]}),
        sub / "part-0000.snappy.parquet",
        compression="snappy",
    )

    # eval_ 前缀文件应被识别为验证集
    (root / "eval_data.json").write_text(
        json.dumps(
            [{"text": "eval json one"}, {"text": "eval json two"}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # valid_ 前缀文件同样应被识别为验证集
    (root / "valid_data.jsonl").write_text(
        '{"text": "valid jsonl one"}\n{"text": "valid jsonl two"}\n',
        encoding="utf-8",
    )
    return root


def test_discover_data_files_recursive(tmp_path):
    """显式指定全部四种格式时应递归发现对应文件。

    只比较文件名而不比较完整路径，避免不同平台（Windows/Linux）路径分隔符差异。
    eval_ / valid_ 文件此时仍会被发现，后续再按前缀做训练/验证分流。
    """
    root = _make_corpus(tmp_path / "corpus")
    files = discover_data_files(str(root), formats=["txt", "json", "jsonl", "parquet"])
    names = {Path(f).name for f in files}
    assert names == {
        "train_data.json",
        "train_data.jsonl",
        "notes.txt",
        "part-0000.snappy.parquet",
        "eval_data.json",
        "valid_data.jsonl",
    }


def test_discover_data_files_parquet_only(tmp_path):
    """formats 限定为 parquet 时只返回 .parquet / .snappy.parquet 文件。

    对应 HF 公开数据集目录同时存在 txt/json/parquet，但只有 parquet 是真实语料的场景，
    避免把 README、dataset_info.json 等非语料文件误加载进来。
    """
    root = _make_corpus(tmp_path / "corpus")
    files = discover_data_files(str(root), formats=["parquet"])
    names = {Path(f).name for f in files}
    assert names == {"part-0000.snappy.parquet"}


def test_discover_data_files_invalid_format(tmp_path):
    """未知格式名应直接抛出 ValueError，避免配置写错后被静默忽略。"""
    with pytest.raises(ValueError):
        discover_data_files(str(tmp_path), formats=["yaml"])


def test_discover_data_files_rejects_single_file_format_mismatch(tmp_path):
    """显式传入单文件时，文件扩展名也必须匹配 data_format。"""
    path = tmp_path / "notes.txt"
    path.write_text("hello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="data_format"):
        discover_data_files(str(path), formats=["json"])


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("eval_data.json", True),
        ("valid_data.jsonl", True),
        ("EVAL_data.json", True),
        ("train_data.json", False),
        ("data.parquet", False),
    ],
)
def test_is_eval_file(name, expected):
    """验证集识别规则：eval_ / valid_ 前缀为验证集，其余为训练集。

    大小写不敏感，因此 EVAL_data.json 也应命中；train_data.json 和普通 parquet
    不属于验证集。
    """
    assert is_eval_file(str(Path(name))) is expected


def test_load_split_mixed_formats(tmp_path):
    """四种格式一次性加载后应合并为一个统一 text 列的数据集。

    txt 2 条 + json 2 条 + jsonl 2 条 + parquet 2 条 = 8 条样本，
    且不同格式自带的额外列（如 parquet 的 schema）应被裁剪掉，只保留 text。
    """
    root = _make_corpus(tmp_path / "corpus")
    ds = load_split(
        [
            str(root / "notes.txt"),
            str(root / "train_data.json"),
            str(root / "train_data.jsonl"),
            str(root / "sub" / "part-0000.snappy.parquet"),
        ]
    )
    # 返回 HF IterableDataset，且所有格式都统一为 text 字段。
    assert isinstance(ds, IterableDataset)
    assert {row["text"] for row in ds} == {
        "txt one",
        "txt two",
        "json one",
        "json two",
        "jsonl one",
        "jsonl two",
        "parquet one",
        "parquet two",
    }


def test_load_split_skips_malformed_jsonl(tmp_path):
    """JSONL 中单行损坏时应跳过并保留其他有效样本。"""
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        '{"text": "good"}\nnot-json\n{"text": 123}\n',
        encoding="utf-8",
    )
    dataset = load_split([str(path)])
    assert [row["text"] for row in dataset] == ["good", "123"]


def test_load_split_skips_bad_json_array_records(tmp_path):
    """JSON 数组中非对象或缺失 text 字段的记录应告警跳过。"""
    path = tmp_path / "mixed.json"
    path.write_text(
        '[{"text": "good"}, {"other": "missing"}, "not-an-object", {"text": 123}]\n',
        encoding="utf-8",
    )
    dataset = load_split([str(path)])
    assert [row["text"] for row in dataset] == ["good", "123"]


class _FakeTokenizer:
    """最小分词器替身。

    返回固定长度的 input_ids / attention_mask，避免测试依赖真实分词器模型；
    load_train_eval_data 内部只要求 tok 能按 batched 方式接收文本列表。
    """

    def __call__(self, texts, **kwargs):
        return {
            "input_ids": [[1, 2] for _ in texts],
            "attention_mask": [[1, 1] for _ in texts],
        }


def test_load_train_eval_data_splits_train_and_eval(tmp_path):
    """端到端验证显式配置全格式时的训练/验证分流与分词结果。

    显式全格式时：训练集 8 条（txt/json/jsonl/parquet 各 2 条），
    验证集 4 条（eval_data.json + valid_data.jsonl 各 2 条）。
    分词后数据集应包含 input_ids 和 attention_mask。
    """
    root = _make_corpus(tmp_path / "corpus")
    cfg = {
        "dataset": {
            "train_data_path": str(root),
            "data_format": ["txt", "json", "jsonl", "parquet"],
        },
        "model": {"max_seq_len": 64},
    }
    train_ds, eval_ds = load_train_eval_data(cfg, _FakeTokenizer())
    train_rows = list(train_ds)
    eval_rows = list(eval_ds)

    # 8 = 2 txt + 2 json + 2 jsonl + 2 parquet
    assert len(train_rows) == 8
    # 4 = 2 eval json + 2 valid jsonl
    assert len(eval_rows) == 4
    assert all("input_ids" in row for row in train_rows)
    assert all("attention_mask" in row for row in train_rows)
    assert all("input_ids" in row for row in eval_rows)
    assert all("attention_mask" in row for row in eval_rows)


@pytest.mark.parametrize("data_format", [None, []])
def test_load_train_eval_data_requires_non_empty_data_format(tmp_path, data_format):
    """配置缺少 data_format 或为空时应直接报错，data_format 必须显式指定。"""
    cfg = {
        "dataset": {"train_data_path": str(tmp_path)},
        "model": {"max_seq_len": 64},
    }
    if data_format is not None:
        cfg["dataset"]["data_format"] = data_format
    with pytest.raises(ValueError, match="data_format"):
        load_train_eval_data(cfg, _FakeTokenizer())


def test_load_train_eval_data_parquet_only(tmp_path):
    """配置 data_format: parquet 时只加载 parquet 语料。

    训练集应只剩 2 条 parquet 样本；目录中虽存在 eval_* / valid_* 的 json/jsonl，
    但格式过滤发生在分流之前，因此验证集为 None。
    """
    root = _make_corpus(tmp_path / "corpus")
    cfg = {
        "dataset": {"train_data_path": str(root), "data_format": "parquet"},
        "model": {"max_seq_len": 64},
    }
    train_ds, eval_ds = load_train_eval_data(cfg, _FakeTokenizer())
    assert len(list(train_ds)) == 2
    assert eval_ds is None


def test_load_train_eval_data_invalid_data_format(tmp_path):
    """配置了未知 data_format 时应在加载阶段直接报错。

    这样用户能尽早发现配置错误，而不是训练到中途才发现数据为空。
    """
    root = _make_corpus(tmp_path / "corpus")
    cfg = {
        "dataset": {"train_data_path": str(root), "data_format": "yaml"},
        "model": {"max_seq_len": 64},
    }
    with pytest.raises(ValueError):
        load_train_eval_data(cfg, _FakeTokenizer())
