"""数据加载与分词器训练共用的文件解析工具。"""

import json
from collections.abc import Iterable

DATA_FORMAT_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "txt": (".txt",),
    "json": (".json",),
    "jsonl": (".jsonl",),
    "parquet": (".parquet",),
}

PARQUET_TEXT_COLUMN_CANDIDATES: tuple[str, ...] = ("text", "content")


def iter_json_array_streaming(
    filepath: str,
    chunk_size: int = 65536,
) -> Iterable[dict]:
    """流式解析 JSON 数组，逐个产出顶层对象。

    不使用 ``json.load()`` 一次性加载整个文件，避免大语料占用过多内存。
    """
    decoder = json.JSONDecoder()
    buffer = ""
    with open(filepath, encoding="utf-8", errors="ignore") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            if len(buffer) == 0 and chunk and chunk[0] == "\ufeff":
                chunk = chunk[1:]
            buffer += chunk
            while True:
                stripped = buffer.lstrip()
                if not stripped:
                    buffer = ""
                    break
                lead = stripped[0]
                if lead in ("[", "]", ","):
                    buffer = stripped[1:]
                    continue
                try:
                    obj, idx = decoder.raw_decode(stripped)
                except json.JSONDecodeError:
                    buffer = stripped
                    break
                buffer = stripped[idx:]
                if isinstance(obj, dict):
                    yield obj

    remaining = buffer.strip()
    if remaining:
        raise ValueError(f"JSON 数组文件存在无法解析的尾部内容: {filepath}")
