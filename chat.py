#!/usr/bin/env python3
"""TarsLM 交互式对话入口。

使用 HF 标准的 tokenizer.apply_chat_template() 格式化对话，
并通过 StoppingCriteria 在生成到 <|end|> 或 <|user|> 时停止。
"""

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from common.checkpoint import find_latest_checkpoint
from common.training_utils import resolve_device
from model.model import TarsLMModel

PROJECT_ROOT = Path(__file__).resolve().parent

# 对话格式控制 token（与 chat_template 和微调数据保持一致）
CHAT_END_TOKEN = "<|end|>"
CHAT_USER_TOKEN = "<|user|>"


class StopOnTokens(StoppingCriteria):
    """HF 标准的停止条件：在生成过程中检测到指定 token ID 时立即停止。

    继承自 transformers.StoppingCriteria，与 model.generate() 原生兼容。
    用于在模型输出 <|end|> 或 <|user|> 时停止（防止模型臆想下一轮用户发言）。
    """

    def __init__(self, stop_token_ids: list):
        """参数:
            stop_token_ids: 停止 token 的 ID 列表，生成到任意一个即停止
        """
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor, **kwargs) -> bool:
        """每生成一步后由框架调用，检查 batch 中第一个样本的最后一个 token。"""
        for stop_id in self.stop_token_ids:
            if input_ids[0, -1] == stop_id:
                return True
        return False


def parse_args():
    """解析命令行参数。"""
    p = argparse.ArgumentParser(description="TarsLM 交互式对话")
    p.add_argument("--checkpoint_dir", type=str, default=str(PROJECT_ROOT / "checkpoints_chat"),
                   help="微调后的 checkpoint 根目录（自动选最新）或具体 checkpoint-n 目录")
    p.add_argument("--tokenizer", type=str, default=None,
                   help="分词器目录；默认从 checkpoint 目录或其上级目录自动推断")
    p.add_argument("--max_new_tokens", type=int, default=128,
                   help="每次回复最多生成的新 token 数量")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="采样温度，<1 更确定，>1 更多样")
    p.add_argument("--top_p", type=float, default=0.95,
                   help="Nucleus 采样累积概率阈值")
    return p.parse_args()


def clean_response(raw_text: str) -> str:
    """从模型原始输出中提取干净的助手回复文本。

    使用 skip_special_tokens=True 解码后,特殊 token 已被 HF 框架滤除，
    此函数只需要处理首尾空白和空回复情况。
    """
    text = raw_text.strip()
    if not text:
        text = "（模型未生成有效回复，请再试一次）"
    return text


def main():
    args = parse_args()

    print("=" * 50)
    print("  TarsLM 对话模式")
    print("  输入 'exit' 或 'quit' 退出对话（每次对话为全新会话）")
    print("=" * 50)

    # ============================================================
    #  第 1 步：定位 checkpoint，再加载分词器与模型
    # ============================================================
    checkpoint_dir = find_latest_checkpoint(args.checkpoint_dir, prefix="[chat]")

    # 具体 checkpoint-n 目录通常没有 tokenizer.json，回退到输出根目录。
    tokenizer_dir = args.tokenizer or checkpoint_dir
    if not os.path.isfile(os.path.join(tokenizer_dir, "tokenizer.json")):
        parent_dir = os.path.dirname(checkpoint_dir)
        if os.path.isfile(os.path.join(parent_dir, "tokenizer.json")):
            tokenizer_dir = parent_dir

    print("[chat] 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    # 验证 chat_template 已就位：缺失时直接退出，避免后续在
    # apply_chat_template() 内部抛出难以理解的模板渲染错误。
    if tokenizer.chat_template is None:
        print(
            "[chat] 错误：加载的分词器未包含 chat_template，无法使用 "
            "apply_chat_template() 格式化对话。\n"
            "      请先运行 python chat_finetune.py 生成微调后的分词器"
            "（默认输出到 checkpoints_chat/），\n"
            "      或通过 --tokenizer 指定包含 chat_template 的分词器目录。"
        )
        raise SystemExit(1)

    # 使用 TarsLMModel.from_pretrained() 恢复完整模型
    # —— HF 标准接口，框架自动读取 config.json + model.safetensors
    print("[chat] 加载模型...")
    device = resolve_device()
    model = TarsLMModel.from_pretrained(checkpoint_dir)
    model.to(device)
    model.eval()

    if args.max_new_tokens >= model.config.max_seq_len:
        raise ValueError(
            f"--max_new_tokens ({args.max_new_tokens}) 必须小于模型 max_seq_len "
            f"({model.config.max_seq_len})"
        )

    # ============================================================
    #  第 2 步：获取停止 token ID 并构建 StoppingCriteria
    # ============================================================
    # 获取聊天特殊 token 的 ID
    user_token_id = tokenizer.convert_tokens_to_ids(CHAT_USER_TOKEN)
    end_token_id = tokenizer.convert_tokens_to_ids(CHAT_END_TOKEN)
    eos_token_id = tokenizer.eos_token_id

    # 构建停止条件列表：
    #   - <|end|> — 对话轮次结束，优先停止
    #   - EOS (</s>) — 标准句尾标记
    #   - <|user|> — 防止模型臆想下一轮用户发言
    stop_ids = [eos_token_id]
    if end_token_id != tokenizer.unk_token_id:
        stop_ids.insert(0, end_token_id)
    if user_token_id != tokenizer.unk_token_id:
        stop_ids.append(user_token_id)

    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_ids)])
    eos_list = stop_ids.copy()

    total_params, size_str = model.count_parameters()
    print(f"[chat] 模型参数量: {total_params:,} ({size_str})")
    print("[chat] 使用 tokenizer.apply_chat_template() 格式化对话")
    print(f"[chat] 停止 token ID 列表: {stop_ids}")
    print()

    # ============================================================
    #  第 3 步：交互式对话循环
    # ============================================================
    # 每次对话为全新会话，不携带历史记录

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        # 空输入跳过
        if not user_input:
            continue
        # 退出指令
        if user_input.lower() in ("exit", "quit"):
            print("再见！")
            break
        # 每次对话已是全新会话，无需手动清空
        if user_input.lower() == "clear":
            continue

        # ---- 设置当前用户消息（每轮独立，不累积历史） ----
        messages = [{"role": "user", "content": user_input}]

        # ---- 用 apply_chat_template() 格式化 prompt ----
        # tokenize=False 返回格式化后的字符串，add_generation_prompt=True 在末尾追加 <|assistant|>
        # apply_chat_template 生成对话格式文本，再手动拼接 <s> 开头
        # 训练时 data_loader 默认 add_special_tokens=True，每个序列以 <s> 开头
        # 推理时也保持 <s> 开头，确保位置编码和注意力模式与训练一致
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt = tokenizer.bos_token + prompt
        max_context = model.config.max_seq_len - args.max_new_tokens
        inputs = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
            max_length=max_context,
            truncation=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # ---- 自回归生成 ----
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_k=50,
                top_p=args.top_p,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_list,
                stopping_criteria=stopping_criteria,
            )

        # ---- 解码新生成部分 ----
        # 只解码新生成的 token（去掉 prompt 部分）
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][prompt_len:]
        # 将可能越界的 token ID 替换为 unk
        new_tokens[new_tokens >= len(tokenizer)] = tokenizer.unk_token_id
        # skip_special_tokens=True:
        #   解码时跳过 <|user|> / <|assistant|> / <|end|> 等特殊 token，
        #   避免这些对话控制 token 出现在最终回复文本中。
        raw_reply = tokenizer.decode(new_tokens, skip_special_tokens=True)
        reply = clean_response(raw_reply)
        # 诊断：显示生成了多少 token 以及全部为特殊 token 的情况
        if len(new_tokens) <= 3 and not reply.strip():
            print(f"[诊断] 仅生成 {len(new_tokens)} 个 token，全部为特殊 token")

        print(f"TarsLM: {reply}")
        print()

        # 不再保存助手回复到历史（每次对话全新）


if __name__ == "__main__":
    main()
