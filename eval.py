#!/usr/bin/env python3
"""TarsLM 推理测试入口：加载 checkpoint 与分词器，生成一段文本。"""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from common.checkpoint import find_latest_checkpoint
from common.training_utils import resolve_device
from model.model import TarsLMModel

PROJECT_ROOT = Path(__file__).resolve().parent

# 用法示例：
# python eval.py --prompt "人工智能"
# python eval.py --checkpoint_dir checkpoints_chat --prompt "什么是人工智能？"

def parse_args():
    p = argparse.ArgumentParser(description="TarsLM 推理测试")
    p.add_argument("--checkpoint_dir", type=str, default=str(PROJECT_ROOT / "checkpoints"),
                   help="checkpoints 根目录（自动选择最新）或具体 checkpoint-n 目录")
    p.add_argument("--tokenizer", type=str, default=str(PROJECT_ROOT / "data" / "tokenizer"),
                   help="分词器目录（tokenizer.json + tokenizer_config.json）")
    p.add_argument("--prompt", type=str, default="人工智能",
                   help="生成起始文本")
    return p.parse_args()


def main():
    args = parse_args()

    # 加载分词器（默认从 data/tokenizer/ 恢复，也可通过 --tokenizer 指定外部目录）
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # 加载模型配置和权重（严格模式，避免 config/checkpoint 不匹配被静默忽略）
    checkpoint_dir = find_latest_checkpoint(args.checkpoint_dir, prefix="[eval]")
    device = resolve_device()
    model = TarsLMModel.from_pretrained(checkpoint_dir)
    model.to(device)
    model.eval()

    # 构造输入
    prompt = args.prompt
    max_context = model.config.max_seq_len - 64
    if max_context < 1:
        raise ValueError("模型 max_seq_len 必须大于默认生成长度 64")
    inputs = tokenizer(
        prompt,
        add_special_tokens=True,
        return_tensors="pt",
        max_length=max_context,
        truncation=True,
    )
    input_ids = inputs["input_ids"].to(device)
    # 去掉末尾 EOS——训练时 EOS 标记序列结束，推理时需要模型自己决定何时停止
    if input_ids[0, -1].item() == tokenizer.eos_token_id:
        input_ids = input_ids[:, :-1]

    # 自回归生成
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,  # 输入 token ID 序列（prompt 编码后的结果）
            max_new_tokens=64,  # 最多生成的新 token 数量（不含 prompt 长度）
            do_sample=True,  # 采样解码
            temperature=0.5,  # 温度系数：更低让输出更确定
            top_k=30,  # Top-K 采样：每步只从概率最高的 30 个 token 中抽取
            top_p=0.9,  # Top-P（nucleus）采样：候选 token 集合累积概率达 0.9 即截断
            repetition_penalty=1.2,  # 重复惩罚：>1 抑制已生成 token 再次出现
            no_repeat_ngram_size=3,  # 禁止任何 3-gram 重复
            pad_token_id=tokenizer.pad_token_id,  # 填充 token ID，attention 计算中会被 mask 掉
            eos_token_id=tokenizer.eos_token_id,  # 结束 token ID，生成到此 token 时自动停止
        )

    # 解码输出
    # 将模型可能生成的越界 token ID 替换为 unk，防止 tokenizer 报 IndexError
    output_ids[output_ids >= len(tokenizer)] = tokenizer.unk_token_id
    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"prompt : {prompt}")
    print(f"output : {result}")


if __name__ == "__main__":
    main()
