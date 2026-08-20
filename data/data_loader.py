"""预训练与微调语料加载工具。

本模块负责递归发现 txt/json/jsonl/parquet 语料、统一为 ``text`` 列，
并按 ``eval_*`` / ``valid_*`` 文件名前缀分流训练集与验证集。
"""

import glob
import json
import os
from collections.abc import Iterator

import pyarrow.parquet as pq
from datasets import IterableDataset, concatenate_datasets
from transformers import PreTrainedTokenizerBase

from common.data_io import (
    DATA_FORMAT_EXTENSIONS,
    PARQUET_TEXT_COLUMN_CANDIDATES,
    iter_json_array_streaming,
)


def discover_data_files(
    data_dir: str,
    formats: list[str],
    file_prefix: str | tuple[str, ...] | None = None,
) -> list[str]:
    """递归发现指定格式的语料文件，并稳定排序去重。"""
    if os.path.isfile(data_dir):
        _validate_single_file_format(data_dir, formats)
        return [data_dir]

    unknown = [fmt for fmt in formats if fmt not in DATA_FORMAT_EXTENSIONS]
    if unknown:
        supported = ", ".join(DATA_FORMAT_EXTENSIONS)
        raise ValueError(f"不支持的数据格式: {', '.join(unknown)}，可选: {supported}")

    extensions = [ext for fmt in formats for ext in DATA_FORMAT_EXTENSIONS[fmt]]
    files: list[str] = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(data_dir, "**", f"*{ext}"), recursive=True))

    if file_prefix is not None:
        files = [
            path
            for path in files
            if os.path.basename(path).startswith(file_prefix)
        ]
    return sorted(set(files))


def _validate_single_file_format(path: str, formats: list[str]) -> None:
    """校验命令行显式传入的单文件是否属于配置格式。"""
    valid_extensions = [
        ext for fmt in formats for ext in DATA_FORMAT_EXTENSIONS[fmt]
    ]
    if not any(path.endswith(ext) for ext in valid_extensions):
        supported = ", ".join(valid_extensions)
        raise ValueError(f"文件 {path} 与 data_format 不匹配，仅支持: {supported}")


def is_eval_file(path: str) -> bool:
    """按文件名前缀识别验证集文件。"""
    name = os.path.basename(path).lower()
    return name.startswith(("eval_", "valid_"))


def _has_any_example(dataset: IterableDataset) -> bool:
    """检查流式数据集是否至少包含一个样本。"""
    try:
        next(iter(dataset.take(1)))
    except StopIteration:
        return False
    return True


def load_split(paths: list[str], text_column: str = "text") -> IterableDataset:
    """按扩展名分组流式加载数据，并统一为 ``text`` 列。"""
    if not paths:
        raise ValueError("数据文件列表为空")

    txt_paths = [path for path in paths if path.endswith(".txt")]
    json_paths = [path for path in paths if path.endswith((".json", ".jsonl"))]
    parquet_paths = [path for path in paths if path.endswith(".parquet")]

    parts: list[IterableDataset] = []
    if txt_paths:
        parts.append(_load_text_paths(txt_paths))
    if json_paths:
        parts.append(_load_json_paths(json_paths, text_column=text_column))
    if parquet_paths:
        parts.append(_load_parquet_paths(parquet_paths, text_column=text_column))

    if not parts:
        raise ValueError("没有可加载的数据文件")
    if not all(_has_any_example(part) for part in parts):
        raise ValueError("数据文件中没有解析到任何有效样本")
    return concatenate_datasets(parts)


def _iter_text_rows(paths: list[str]) -> Iterator[dict]:
    """逐文件逐行读取纯文本，空行不参与训练。"""
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                text = line.strip()
                if text:
                    yield {"text": text}


def _load_text_paths(paths: list[str]) -> IterableDataset:
    """把纯文本文件包装为流式 IterableDataset。"""
    return IterableDataset.from_generator(
        _iter_text_rows,
        gen_kwargs={"paths": paths},
    )


def _iter_json_rows(paths: list[str], text_column: str) -> Iterator[dict]:
    """流式读取 JSON 数组或 JSONL 文件，统一为 ``text`` 字段。"""
    for path in paths:
        if path.endswith(".jsonl"):
            yield from _iter_jsonl_rows(path, text_column)
        else:
            for record_index, item in enumerate(
                iter_json_array_streaming(path),
                start=1,
            ):
                try:
                    yield _normalize_json_row(item, path, text_column)
                except (TypeError, ValueError) as exc:
                    print(
                        f"警告：跳过无法使用的 JSON 数组记录: "
                        f"{path}:{record_index}: {exc}"
                    )


def _load_json_paths(paths: list[str], text_column: str) -> IterableDataset:
    """把 JSON/JSONL 文件包装为流式 IterableDataset。"""
    return IterableDataset.from_generator(
        _iter_json_rows,
        gen_kwargs={"paths": paths, "text_column": text_column},
    )


def _iter_jsonl_rows(path: str, text_column: str) -> Iterator[dict]:
    """逐行解析 JSONL 文件；单行损坏时告警并跳过。"""
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"警告：跳过无法解析的 JSONL 行: {path}:{line_number}")
                continue
            try:
                yield _normalize_json_row(item, path, text_column)
            except (TypeError, ValueError) as exc:
                print(
                    f"警告：跳过无法使用的 JSONL 行: "
                    f"{path}:{line_number}: {exc}"
                )


def _iter_parquet_rows(paths: list[str], text_column: str) -> Iterator[dict]:
    """按批读取 Parquet 的文本列，避免一次性加载整张表。"""
    for path in paths:
        with pq.ParquetFile(path) as parquet_file:
            column_names = parquet_file.schema.names
            if text_column not in column_names:
                raise ValueError(
                    f"parquet 文件缺少 {text_column} 列: {column_names}"
                )
            batches = parquet_file.iter_batches(
                batch_size=100_000,
                columns=[text_column],
            )
            for batch in batches:
                for value in batch.column(text_column).to_pylist():
                    text = str(value).strip() if value is not None else ""
                    if text:
                        yield {"text": text}


def _load_parquet_paths(
    paths: list[str],
    text_column: str,
) -> IterableDataset:
    """把 Parquet 文件包装为流式 IterableDataset。"""
    return IterableDataset.from_generator(
        _iter_parquet_rows,
        gen_kwargs={"paths": paths, "text_column": text_column},
    )


def _normalize_json_row(item: object, path: str, text_column: str) -> dict:
    """校验并提取单个文本字段。"""
    if not isinstance(item, dict):
        raise TypeError("记录不是 JSON 对象")
    if text_column not in item:
        raise ValueError(f"缺少列 {text_column}: {list(item.keys())}")
    value = item[text_column]
    text = str(value).strip() if value is not None else ""
    return {"text": text}


def _tokenize_examples(
    examples: dict,
    tok: PreTrainedTokenizerBase,
    max_seq_len: int,
) -> dict:
    """对一批文本做截断分词，返回可供 Trainer 消费的字段。"""
    return tok(examples["text"], max_length=max_seq_len, truncation=True)


def _tokenize_dataset(
    dataset: IterableDataset,
    tok: PreTrainedTokenizerBase,
    max_seq_len: int,
) -> IterableDataset:
    """延迟分词并移除原始文本列，避免 Trainer 侧缓存整份语料。"""
    return dataset.map(
        _tokenize_examples,
        batched=True,
        remove_columns=["text"],
        fn_kwargs={"tok": tok, "max_seq_len": max_seq_len},
    )


def load_train_eval_data(
    cfg: dict,
    tok: PreTrainedTokenizerBase,
    data_dir: str | None = None,
    file_prefix: str | tuple[str, ...] | None = None,
    seed: int = 42,
) -> tuple[IterableDataset, IterableDataset | None]:
    """扫描语料目录，按配置加载并分词训练/验证数据集。"""
    dc = cfg["dataset"]
    max_seq_len = cfg["model"]["max_seq_len"]
    data_dir = data_dir or dc["train_data_path"]

    data_format = dc.get("data_format")
    if data_format is None:
        raise ValueError(
            "配置缺少 dataset.data_format，必须显式指定语料格式（txt/json/jsonl/parquet）"
        )
    formats = [data_format] if isinstance(data_format, str) else list(data_format)
    if not formats:
        raise ValueError("dataset.data_format 不能为空，必须显式指定语料格式")

    data_files = discover_data_files(
        data_dir,
        formats=formats,
        file_prefix=file_prefix,
    )
    if not data_files:
        format_desc = ", ".join(
            ext for fmt in formats for ext in DATA_FORMAT_EXTENSIONS[fmt]
        )
        raise FileNotFoundError(f"在 {data_dir} 下未找到 {format_desc} 文件")

    if os.path.isfile(data_dir):
        train_files = data_files
        eval_files: list[str] = []
    else:
        train_files = [path for path in data_files if not is_eval_file(path)]
        eval_files = [path for path in data_files if is_eval_file(path)]

    if not train_files:
        raise FileNotFoundError(f"在 {data_dir} 下未找到训练数据文件")

    print(f"发现 {len(train_files)} 个训练数据文件、{len(eval_files)} 个验证数据文件")
    for path in train_files:
        print(f"  训练数据: {path.replace(os.sep, '/')}")
    for path in eval_files:
        print(f"  验证数据: {path.replace(os.sep, '/')}")

    text_column = dc.get("text_column")
    if text_column is None:
        parquet_files = [path for path in train_files if path.endswith(".parquet")]
        if parquet_files:
            with pq.ParquetFile(parquet_files[0]) as parquet_file:
                schema_names = parquet_file.schema.names
            for candidate in PARQUET_TEXT_COLUMN_CANDIDATES:
                if candidate in schema_names:
                    text_column = candidate
                    break
            if text_column is None:
                raise ValueError(f"parquet 文件缺少 text/content 列: {schema_names}")
        else:
            text_column = "text"

    print(f"  文本列名: {text_column}")
    train_dataset = load_split(train_files, text_column=text_column)
    eval_dataset = (
        load_split(eval_files, text_column=text_column)
        if eval_files
        else None
    )
    if eval_dataset is not None:
        print("加载验证数据完成")
    else:
        print("未发现 eval_* / valid_* 验证文件，已跳过评估")

    train_dataset = _tokenize_dataset(train_dataset, tok, max_seq_len)
    if not _has_any_example(train_dataset):
        raise ValueError(f"在 {data_dir} 下没有解析到任何有效训练样本")
    train_dataset = train_dataset.shuffle(seed=seed, buffer_size=10_000)
    if cfg.get("training", {}).get("max_steps"):
        train_dataset = train_dataset.repeat(None)
    if eval_dataset is not None:
        eval_dataset = _tokenize_dataset(eval_dataset, tok, max_seq_len)
        if not _has_any_example(eval_dataset):
            raise ValueError(f"在 {data_dir} 下没有解析到任何有效验证样本")
    return train_dataset, eval_dataset
