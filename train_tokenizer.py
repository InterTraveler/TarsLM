#!/usr/bin/env python3
"""TarsLM Hugging Face fast tokenizer 训练脚本。

本脚本负责把预训练语料统一收集为「一行一句」的纯文本，再用 Hugging Face
``tokenizers`` 库训练 BPE / unigram 分词器，最后包装成
``PreTrainedTokenizerFast`` 并保存到指定目录。保存结果可通过
``AutoTokenizer.from_pretrained("./data/tokenizer/")`` 直接加载，
供预训练、对话微调和推理统一使用。

快速上手（使用 JSON 语料训练 8000 词表并输出到 ./data/tokenizer/）：

    python train_tokenizer.py --data_format json --vocab_size 8000 --corpus ./data/train_data/ --output ./data/tokenizer/

常用参数：
    --data_format  语料格式，必填，可选 txt / json / jsonl / parquet
    --corpus       语料目录或单文件路径，目录会递归加载指定格式文件
    --vocab_size   目标词表大小
    --output       分词器输出目录
    --model_type   分词算法：unigram（默认）/ bpe
"""

import argparse
import ctypes
import glob
import json
import os
import sys
from collections.abc import Iterable, Sequence

import pyarrow.parquet as pq
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    pre_tokenizers,
    processors,
    trainers,
)
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from common.data_io import (
    DATA_FORMAT_EXTENSIONS,
    PARQUET_TEXT_COLUMN_CANDIDATES,
    iter_json_array_streaming,
)
from data.data_loader import is_eval_file

if os.name == "nt":
    ctypes.windll.kernel32.SetConsoleOutputCP(65001)  # pyright: ignore[reportAttributeAccessIssue]
sys.stdout.reconfigure(encoding="utf-8")

# 训练时统一预留基础特殊 token 与聊天控制 token。
# 聊天控制 token 不能作为普通子词学习，否则微调阶段会出现部分 embedding
# 来自预训练子词、部分来自随机初始化，导致初始化不可复现。
SPECIAL_TOKENS: tuple[str, ...] = (
    "<pad>",
    "<s>",
    "</s>",
    "<unk>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
)

CHAT_CONTROL_TOKENS: tuple[str, ...] = (
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
)


def _in_pycharm() -> bool:
    """检测当前是否运行在 PyCharm 中。

    PyCharm 终端对 tqdm 的进度条支持不佳，会输出大量控制字符，
    因此运行在 PyCharm 时统一关闭进度条，让日志更干净。
    """
    return bool(os.environ.get("PYCHARM_HOSTED"))


def _resolve_data_format(data_format: str) -> tuple[str, ...]:
    """校验 --data_format，并返回该格式对应的文件扩展名集合。"""
    if data_format not in DATA_FORMAT_EXTENSIONS:
        supported = ", ".join(DATA_FORMAT_EXTENSIONS)
        raise ValueError(f"不支持的数据格式: {data_format}，可选: {supported}")
    return DATA_FORMAT_EXTENSIONS[data_format]


def _parquet_text_column(path: str, text_column: str | None) -> str:
    """返回 parquet 文件中的文本列名。

    显式传入 text_column 时直接使用；否则在 text/content 中自动选择。
    """
    if text_column:
        return text_column
    with pq.ParquetFile(path) as parquet_file:
        column_names = parquet_file.schema.names
    for candidate in PARQUET_TEXT_COLUMN_CANDIDATES:
        if candidate in column_names:
            return candidate
    raise ValueError(f"parquet 文件缺少 text/content 列: {column_names}")


def _iter_parquet_texts(
    path: str, text_column: str, progress: bool = True
) -> Iterable[str]:
    """按批读取 parquet 的文本列，产出去除首尾空白后的非空文本行。

    parquet 是列式二进制格式，不能按文本行读取，因此用 pyarrow 的
    ``iter_batches()`` 分批流式读取，控制内存占用。
    """
    with pq.ParquetFile(path) as parquet_file:
        column_names = parquet_file.schema.names
        if text_column not in column_names:
            raise ValueError(f"parquet 文件缺少 {text_column} 列: {column_names}")
        batch_iter = parquet_file.iter_batches(batch_size=100_000, columns=[text_column])
        if progress and not _in_pycharm():
            batch_iter = tqdm(batch_iter, desc="  读取 parquet", unit="batch", leave=False)
        for batch in batch_iter:
            for value in batch.column(text_column).to_pylist():
                if value is None:
                    continue
                text = str(value).strip()
                if text:
                    yield text


def collect_texts(
    corpus_path: str,
    output_file: str,
    text_column: str | None = None,
    *,
    append: bool = False,
    data_format: str,
    no_progress: bool = False,
) -> int:
    """按 data_format 收集语料文本，合并写入一个纯文本文件。

    支持两种输入：
    - 单文件：corpus_path 是具体文件，扩展名必须匹配 data_format；
    - 目录：递归搜索 data_format 对应扩展名的所有文件并排序去重。

    文本提取规则：
    - txt：每行一句，直接写入；
    - jsonl：每行一个 JSON，优先取 "text" 字段，解析失败保留原始行；
    - json：整体是 JSON 数组，流式解析后逐个取 "text" 字段；
    - parquet：读取 text/content 列（可显式指定 text_column），按批流式读取。

    返回本次实际写入的有效文本行数；所有文件都提取不到文本时抛出 RuntimeError。
    """
    # 第 1 步：确定待处理文件列表。单文件直接使用，目录递归搜索并排序去重。
    extensions = _resolve_data_format(data_format)
    if os.path.isfile(corpus_path):
        if not corpus_path.endswith(extensions):
            raise ValueError(
                f"文件 {corpus_path} 不属于 data_format={data_format} 指定格式"
            )
        files = [corpus_path]
    else:
        files = []
        for ext in extensions:
            files.extend(
                glob.glob(os.path.join(corpus_path, "**", f"*{ext}"), recursive=True)
            )
        files = sorted(set(files))
        files = [path for path in files if not is_eval_file(path)]

    # 第 2 步：找不到任何文件时尽早报错，避免后续静默训练出空模型。
    if not files:
        format_desc = ", ".join(extensions)
        raise FileNotFoundError(
            f"在 {corpus_path} 下未找到 {format_desc} 文件（data_format={data_format}）。"
            "请确认语料格式为以下之一："
            "  - .txt    ：纯文本，每行一句"
            "  - .jsonl  ：每行一个 JSON，含 'text' 字段"
            "  - .json   ：JSON 数组 [{'text': ...}, ...]"
            "  - .parquet：含 'text' 或 'content' 列"
        )

    count = 0  # 有效文本行计数器。

    # 第 3 步：逐文件解析并写入临时合并文件，统一使用 UTF-8。
    open_mode = "a" if append else "w"
    with open(output_file, open_mode, encoding="utf-8") as out:
        show_progress = not no_progress and not _in_pycharm()
        file_iter = (
            tqdm(
                sorted(files),
                desc="收集语料",
                unit="file",
                bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}]",
            )
            if show_progress
            else sorted(files)
        )
        for fp in file_iter:
            # parquet 是列式二进制格式，需要走 pyarrow 按批读取分支。
            if fp.endswith(".parquet"):
                text_column_name = _parquet_text_column(fp, text_column)
                for text in _iter_parquet_texts(fp, text_column_name):
                    out.write(text + "\n")
                    count += 1
                continue

            # errors="ignore"：遇到无法解码的字节跳过，保证大语料能继续处理。
            with open(fp, encoding="utf-8", errors="ignore") as f:
                # .json（非 .jsonl）走流式数组解析，避免大文件 MemoryError。
                if fp.endswith(".json") and not fp.endswith(".jsonl"):
                    for item in iter_json_array_streaming(fp):
                        raw_text = item.get("text", "")
                        text = str(raw_text).strip() if raw_text is not None else ""
                        if text:
                            out.write(text + "\n")
                            count += 1
                else:
                    # .txt / .jsonl 逐行读取：txt 取整行，jsonl 优先取 "text"。
                    for line in f:
                        line = line.strip()  # 去掉行首尾空白，避免生成无意义 token。
                        if not line:  # 空行没有语义信息，直接跳过。
                            continue
                        if fp.endswith(".jsonl"):
                            try:
                                item = json.loads(line)
                                # 只有 JSON 对象才提取 "text" 字段；合法 JSON 但非对象的行
                                # （字符串/数组/数字等）按普通文本行保留，避免 AttributeError。
                                if isinstance(item, dict):
                                    raw_text = item.get("text")
                                    line = str(raw_text) if raw_text is not None else line
                            except json.JSONDecodeError:
                                # 该行可能本来就是普通文本，保留原行继续处理。
                                pass
                        out.write(line + "\n")
                        count += 1

    # 第 4 步：健壮性检查，所有文件都解析不出文本时提前终止。
    if count == 0:
        raise RuntimeError(
            f"已读取 {len(files)} 个文件，但未提取到任何有效文本行。"
            "请检查语料文件内容是否为空或格式是否正确。"
        )
    return count


def _build_tokenizer(model_type: str, vocab_size: int, show_progress: bool):
    """构建 Hugging Face Tokenizer 及其对应的 Trainer。

    BPE 使用 ByteLevel 预分词/解码器，字节级回退可以处理任意 Unicode 文本；
    unigram 使用 Metaspace（``▁`` 表示空格）作为预分词/解码器。
    实际传给 tokenizers 的词表大小会先扣掉全部特殊 token，保存后
    ``len(tokenizer)`` 恰好等于用户传入的 vocab_size。
    """
    # 词表至少要能放下所有特殊 token，否则无法完成后续包装。
    if vocab_size <= len(SPECIAL_TOKENS):
        raise ValueError(f"vocab_size 必须大于特殊 token 数量 {len(SPECIAL_TOKENS)}")
    backend_vocab_size = vocab_size - len(SPECIAL_TOKENS)

    if model_type == "bpe":
        # BPE 初始字母表只包含字节表；特殊 token 在训练完成后统一追加，
        # 避免把聊天控制 token 当成普通子词参与 BPE 合并。
        initial_alphabet = list(pre_tokenizers.ByteLevel.alphabet())
        tokenizer = Tokenizer(
            models.BPE(
                vocab={token: i for i, token in enumerate(initial_alphabet)},
                merges=[],
                unk_token="<unk>",
            )
        )
        # ByteLevel 预分词以字节为单位处理文本，decoder 负责还原空格与 Unicode。
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(
            vocab_size=backend_vocab_size,
            special_tokens=[],
            initial_alphabet=initial_alphabet,
            show_progress=show_progress,
        )
        tokenizer.decoder = decoders.ByteLevel()
        return tokenizer, trainer

    # unigram 分支：预置特殊 token 并指定 <unk> 的 id，再交给 UnigramTrainer 训练。
    tokenizer = Tokenizer(
        models.Unigram(
            vocab=[(token, 0.0) for token in SPECIAL_TOKENS],
            unk_id=SPECIAL_TOKENS.index("<unk>"),
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement="\u2581", prepend_scheme="always"
    )
    tokenizer.add_special_tokens(list(SPECIAL_TOKENS))
    trainer = trainers.UnigramTrainer(
        # UnigramTrainer 会把 <unk> 从目标词表中扣除，这里补回 1，保证最终
        # ``len(tokenizer) == vocab_size``。
        vocab_size=backend_vocab_size + 1,
        special_tokens=[],
        unk_token="<unk>",
        show_progress=show_progress,
    )
    tokenizer.decoder = decoders.Metaspace(
        replacement="\u2581", prepend_scheme="always"
    )
    return tokenizer, trainer


def train_tokenizer(
    corpus_path: str | None = None,
    output_dir: str = "./data/tokenizer/",
    vocab_size: int = 32000,
    model_type: str = "unigram",
    text_column: str | None = None,
    no_progress: bool = False,
    *,
    corpus_paths: Sequence[str] | None = None,
    data_format: str,
) -> None:
    """训练 fast tokenizer 并以 HuggingFace 兼容格式保存。

    参数:
        corpus_path: 单语料路径（目录或文件）；与 ``corpus_paths`` 至少提供一个，传入 ``corpus_paths`` 时优先使用多语料。
        output_dir: 分词器保存目录，默认 ``./data/tokenizer/``。
        corpus_paths: 可选多语料路径列表，传入时优先于 ``corpus_path``。

    执行流程：
    1. collect_texts() 把语料合并成一行一句的临时纯文本；
    2. _build_tokenizer() 按 model_type 构建 BPE / unigram 并训练；
    3. 包装为 PreTrainedTokenizerFast，配置 bos/eos/unk/pad；
    4. 设置单句/双句模板并 save_pretrained() 输出 tokenizer.json；
    5. finally 中清理临时合并文件与旧版 SentencePiece 遗留文件。
    """
    if not corpus_path and not corpus_paths:
        raise ValueError("corpus_path 与 corpus_paths 至少提供一个语料路径")
    # 创建输出目录，目录已存在时不会报错。
    os.makedirs(output_dir, exist_ok=True)
    # 语料合并文件放在输出目录内，训练结束后无论成败都会在 finally 中清理。
    merged_path = os.path.join(output_dir, "_train_corpus.txt")

    corpus_sources = list(corpus_paths) if corpus_paths else [corpus_path]
    total = 0
    for index, source in enumerate(corpus_sources):
        print(f"\n正在汇合语料: {source}")
        try:
            total += collect_texts(
                source,
                merged_path,
                text_column=text_column,
                data_format=data_format,
                no_progress=no_progress,
                append=index > 0,
            )
        except Exception:
            if os.path.isfile(merged_path):
                os.remove(merged_path)
            raise
    print(f"  共收集 {total} 行文本")

    # 按算法构建 tokenizer + trainer；vocab_size 会在内部扣掉特殊 token。
    tokenizer, trainer = _build_tokenizer(
        model_type=model_type,
        vocab_size=vocab_size,
        show_progress=not no_progress and not _in_pycharm(),
    )
    print(f"\n开始训练 Hugging Face {model_type} tokenizer (vocab_size={vocab_size})...")
    try:
        # tokenizers 直接接受文本文件路径列表，内部会流式读取训练样本。
        tokenizer.train([merged_path], trainer=trainer)
        # BPE 需要在训练后追加特殊 token，确保控制 token 落在普通子词区间之外。
        tokenizer.add_special_tokens(list(SPECIAL_TOKENS))
    finally:
        if os.path.isfile(merged_path):
            os.remove(merged_path)

    # 包装成 HF fast tokenizer，并显式声明 bos/eos/unk/pad 四个特殊 token。
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    # 设置 HF 模板：单句自动包成 <s> $A </s>，双句包成 <s> $A </s> <s> $B </s>。
    fast_tokenizer.backend_tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> <s> $B </s>",
        special_tokens=[
            ("<s>", fast_tokenizer.bos_token_id or 1),
            ("</s>", fast_tokenizer.eos_token_id or 2),
        ],
    )
    fast_tokenizer.save_pretrained(output_dir)
    # 清理旧版 SentencePiece / 慢速分词器产物，避免目录里残留过期文件。
    for legacy_name in ("tokenizer.model", "tokenizer.vocab", "added_tokens.json"):
        legacy_path = os.path.join(output_dir, legacy_name)
        if os.path.isfile(legacy_path):
            os.remove(legacy_path)
    print(f"\n分词器已保存至 {output_dir} (词表大小: {len(fast_tokenizer)})")


def main():
    """命令行入口：解析参数后调用 train_tokenizer() 完成训练与保存。

    快速上手：
      python train_tokenizer.py --data_format json --vocab_size 8000 --corpus ./data/train_data/ --output ./data/tokenizer/
    """
    parser = argparse.ArgumentParser(
        description="TarsLM Hugging Face fast tokenizer 训练工具"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="./data/train_data/",
        help="预训练语料目录路径（递归加载 --data_format 指定格式的文件）",
    )
    parser.add_argument(
        "--corpus_paths",
        type=str,
        nargs="+",
        default=None,
        help="多个语料目录或单文件路径，空格分隔；传入后优先于 --corpus",
    )
    parser.add_argument(
        "--data_format",
        type=str,
        required=True,
        choices=list(DATA_FORMAT_EXTENSIONS),
        help="语料格式，必填；可选 txt/json/jsonl/parquet",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/tokenizer/",
        help="分词器输出目录",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=8000,
        help="目标词表大小",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="unigram",
        choices=["unigram", "bpe"],
        help="分词算法：unigram（默认）/bpe",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="parquet 文件中的文本列名（默认自动识别 text/content）",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="禁用进度条",
    )
    args = parser.parse_args()

    train_tokenizer(
        corpus_path=args.corpus,
        corpus_paths=args.corpus_paths,
        output_dir=args.output,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        text_column=args.text_column,
        no_progress=args.no_progress,
        data_format=args.data_format,
    )


if __name__ == "__main__":
    main()
