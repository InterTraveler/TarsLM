#!/usr/bin/env python3
"""TarsLM 对话微调入口。

加载分词器 → 注册聊天特殊 token 与 chat_template → 从基础 checkpoint
恢复模型并调整词嵌入 → 加载对话语料 → 使用 HF Trainer 继续 CLM 训练。
"""

import argparse
from pathlib import Path

import torch
import yaml
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

from common.checkpoint import find_latest_checkpoint
from common.training_utils import (
    apply_finetune_overrides,
    resolve_config_paths,
    resolve_deepspeed_config_path,
    validate_training_config,
)
from data.data_loader import load_train_eval_data
from model.model import TarsLMModel
from train_tokenizer import CHAT_CONTROL_TOKENS

# 聊天场景专用特殊 token 列表
# <|user|>  — 用户发言起始标记
# <|assistant|> — 助手回复起始标记，模型在生成时以此为信号开始产出回复
# <|end|> — 对话轮次结束标记，模型在生成时以此为信号停止产出
CHAT_SPECIAL_TOKENS = list(CHAT_CONTROL_TOKENS)

# Jinja2 对话模板字符串（HF 标准：tokenizer.apply_chat_template() 使用）
# 模板在保存到 tokenizer_config.json 后，可由 apply_chat_template(messages, tokenize=False) 调用
# 输入 messages 格式: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
# 输出格式: <|user|>...<|assistant|>...<|end|>\n<|user|>...<|assistant|>
# 当 add_generation_prompt=True 时，末尾追加 <|assistant|> 等待模型续写
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|{{ message['role'] }}|>{{ message['content'] }}"
    "{% if message['role'] == 'assistant' %}<|end|>\n{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def parse_args():
    """解析对话微调入口的命令行参数。"""
    p = argparse.ArgumentParser(description="TarsLM 对话微调")
    p.add_argument("--config", type=str, default="config/default.yaml",
                   help="配置文件路径")
    p.add_argument("--max_steps", type=int, default=None,
                   help="覆盖配置文件中的最大训练步数")
    p.add_argument("--resume", type=str, default=None,
                   help="恢复微调：latest/true=自动找最新 checkpoint，或指定路径")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    return p.parse_args()


def load_config(path: str) -> dict:
    """从 YAML 配置文件加载全局配置字典。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_resume_checkpoint(resume: str | None, output_dir: str) -> str | None:
    """把 ``latest`` 形式的续训参数解析为具体 checkpoint 目录。"""
    if resume is None:
        return None
    if resume.lower() in ("latest", "true"):
        return find_latest_checkpoint(output_dir, prefix="[chat]")
    return resume


def _validate_chat_control_tokens(tokenizer: PreTrainedTokenizerBase) -> None:
    """确保聊天控制 token 全部位于预留 ID 区间。

    旧分词器可能把 ``<|assistant|>`` 这类字面串学成普通子词；若继续微调，
    会复用预训练 embedding，而另两个控制 token 却是随机初始化。这里显式检查，
    防止产生不可复现的 embedding 初始化。
    """
    added_vocab = tokenizer.get_added_vocab()
    normal_vocab_size = len(tokenizer) - len(added_vocab)
    for token in CHAT_SPECIAL_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == tokenizer.unk_token_id or token_id < normal_vocab_size:
            raise ValueError(
                f"聊天控制 token {token} 未落在预留 ID 区间 "
                f"(id={token_id}, normal_vocab_size={normal_vocab_size})。"
                "请使用修复后的 train_tokenizer.py 重新训练分词器。"
            )


def _validate_tokenizer_vocab_size(
    tokenizer_vocab_size: int,
    model_vocab_size: int,
) -> None:
    """校验微调分词器不能小于 checkpoint 模型的词表。"""
    if tokenizer_vocab_size < model_vocab_size:
        raise ValueError(
            f"分词器词表大小 ({tokenizer_vocab_size}) 小于 checkpoint 模型词表大小 "
            f"({model_vocab_size})，无法安全映射词嵌入。"
            "请使用与 checkpoint 匹配的分词器，或重新训练分词器后再继续微调。"
        )


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    project_root = Path(__file__).resolve().parent
    cfg = load_config(str(config_path))

    # 先合并 pretrain 默认值，再叠加 finetune 覆盖值，最后应用命令行覆盖
    cfg = apply_finetune_overrides(cfg)
    cfg = resolve_config_paths(cfg, project_root)
    validate_training_config(cfg)
    set_seed(args.seed)
    if args.max_steps is not None:
        cfg["training"]["max_steps"] = args.max_steps

    fc = cfg.get("finetune", {})

    if torch.cuda.is_available():
        print(f"设备: CUDA - {torch.cuda.get_device_name(0)}")
    else:
        print("设备: CPU")

    # ============================================================
    #  加载分词器，添加聊天特殊 token，设置 chat_template
    # ============================================================
    print("[chat] 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["dataset"]["tokenizer_path"])
    old_vocab_size = len(tokenizer)
    print(f"[chat] 原始词表大小: {old_vocab_size}")

    # 新分词器训练阶段已预留控制 token；旧分词器缺少数值时会在这里补注册。
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": CHAT_SPECIAL_TOKENS}
    )
    if num_added:
        print(f"[chat] 已补注册 {num_added} 个聊天特殊 token:")
    else:
        print("[chat] 聊天特殊 token 已由分词器预留:")
    for t in CHAT_SPECIAL_TOKENS:
        tid = tokenizer.convert_tokens_to_ids(t)
        print(f"  {t} -> id={tid}")
    _validate_chat_control_tokens(tokenizer)

    # 设置 chat_template（内存中），训练完成后 save_chat_template() 负责持久化
    tokenizer.chat_template = CHAT_TEMPLATE
    print("[chat] chat_template 已在内存中设置")

    # ============================================================
    #  从基础 checkpoint 恢复模型，续训时优先使用微调 checkpoint
    # ============================================================
    pretrained_ckpt = fc.get("pretrained_checkpoint", "./checkpoints/")
    output_dir = cfg["paths"]["checkpoint_dir"]
    resume_checkpoint = resolve_resume_checkpoint(args.resume, output_dir)
    base_checkpoint = resume_checkpoint or find_latest_checkpoint(
        pretrained_ckpt,
        prefix="[chat]",
    )
    print(f"[chat] 从 checkpoint 加载模型: {base_checkpoint}")
    model = TarsLMModel.from_pretrained(base_checkpoint)

    model_vocab_size = model.config.vocab_size
    _validate_tokenizer_vocab_size(len(tokenizer), model_vocab_size)
    if len(tokenizer) > model_vocab_size:
        model.resize_token_embeddings(len(tokenizer))
        print(f"[chat] 词表大小: {model_vocab_size} -> {len(tokenizer)}")

    total_params, size_str = model.count_parameters()
    print(f"[chat] 模型参数量: {total_params:,} ({size_str})")

    # ============================================================
    #  加载对话数据
    #  加载 finetune 配置指定的对话语料目录
    # ============================================================
    print("[chat] 加载对话数据...")
    train_dataset, eval_dataset = load_train_eval_data(
        cfg,
        tokenizer,
        seed=args.seed,
    )

    # ============================================================
    #  配置 HF Trainer 训练参数
    # ============================================================
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    tc, hc, pc = cfg["training"], cfg["hardware"], cfg["paths"]

    fp16 = tc["use_mixed_precision"] and tc["precision"] == "fp16"
    bf16 = tc["use_mixed_precision"] and tc["precision"] == "bf16"
    es = "steps" if eval_dataset is not None else "no"

    ta = TrainingArguments(
        output_dir=pc["checkpoint_dir"],
        per_device_train_batch_size=tc["batch_size"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        learning_rate=tc["learning_rate"],
        warmup_steps=tc["warmup_steps"],
        max_steps=tc["max_steps"],
        lr_scheduler_type=tc["lr_scheduler"],
        weight_decay=tc["weight_decay"],
        adam_beta1=tc["adam_beta1"],
        adam_beta2=tc["adam_beta2"],
        adam_epsilon=tc["adam_epsilon"],
        max_grad_norm=tc["max_grad_norm"],
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=hc.get("enable_gradient_checkpointing", False),
        logging_steps=tc["log_interval"],
        save_steps=tc["save_interval"],
        eval_steps=tc["eval_interval"],
        eval_strategy=es,
        save_total_limit=tc["keep_checkpoint_max"],
        report_to="tensorboard",
        deepspeed=resolve_deepspeed_config_path(cfg, project_root),
        dataloader_num_workers=cfg["dataset"].get("num_workers", 0),
        dataloader_pin_memory=True,
        remove_unused_columns=True,
        load_best_model_at_end=False,
    )

    # ============================================================
    #  构建 Trainer 并启动微调训练
    # ============================================================
    trainer = Trainer(
        model=model, args=ta,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=collator,
    )

    print("[chat] 开始对话微调...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    print(f"[chat] 微调完成，checkpoint 已保存至 {pc['checkpoint_dir']}")

    # 保存分词器（包含新增的聊天 token）
    tokenizer.save_pretrained(pc["checkpoint_dir"])
    print(f"[chat] 分词器（含 chat_template）已保存至 {pc['checkpoint_dir']}")


if __name__ == "__main__":
    main()
