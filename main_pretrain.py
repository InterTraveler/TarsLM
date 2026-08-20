#!/usr/bin/env python3
"""TarsLM 预训练入口。

加载 YAML 配置 → 分词器 → 模型 → 数据集 → HF Trainer 开始预训练，
支持通过命令行覆盖配置、恢复 checkpoint 或指定语料路径。
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

from common.training_utils import (
    merge_pretrain_sections,
    resolve_config_paths,
    resolve_deepspeed_config_path,
    validate_training_config,
)
from data.data_loader import load_train_eval_data
from model.config import TarsLMConfig
from model.model import TarsLMModel


def parse_args():
    """解析预训练入口的命令行参数。"""
    p = argparse.ArgumentParser(description="TarsLM 预训练启动脚本")
    p.add_argument("--config", type=str, default="config/default.yaml")  # 默认配置文件
    p.add_argument("--max_steps", type=int, default=None)  # 覆盖最大训练步数
    p.add_argument("--resume", type=str, default=None,
                   help="恢复训练：latest/true=自动找最新 checkpoint，或指定路径")
    p.add_argument("--train_data_path", type=str, default=None,
                   help="预训练语料目录或文件路径，目录时递归加载 data_format 指定格式的文件")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    return p.parse_args()


def load_config(path: str) -> dict:
    """从 YAML 文件加载全局配置字典。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model_config(cfg: dict, tokenizer: PreTrainedTokenizerBase) -> TarsLMConfig:
    """从 YAML 配置中提取模型架构参数，其余使用 TarsLMConfig 默认值。"""
    mc = cfg["model"]  # 模型架构段
    moe_c = cfg["moe"]  # MoE 配置段
    tokenizer_vocab_size = len(tokenizer)
    configured_vocab_size = mc.get("vocab_size")
    if configured_vocab_size is not None and configured_vocab_size != tokenizer_vocab_size:
        raise ValueError(
            f"model.vocab_size ({configured_vocab_size}) 与分词器词表大小 "
            f"({tokenizer_vocab_size}) 不一致"
        )

    return TarsLMConfig(
        vocab_size=tokenizer_vocab_size,  # 已通过上面的校验，确保与分词器一致
        hidden_size=mc["hidden_size"],
        num_layers=mc["num_layers"],
        num_attention_heads=mc["num_attention_heads"],
        intermediate_size=mc["intermediate_size"],
        max_seq_len=mc["max_seq_len"],
        num_key_value_heads=mc.get("num_key_value_heads", mc["num_attention_heads"]),
        norm_eps=mc.get("norm_eps", 1e-6),
        attention_dropout=mc.get("attention_dropout", 0.0),
        hidden_dropout=mc.get("hidden_dropout", 0.1),
        rope_theta=mc.get("rope_theta", 10000.0),
        use_moe=moe_c.get("enabled", True),  # YAML 未指定则默认开启 MoE
        num_experts=moe_c.get("num_experts", 2),  # 默认 2 个专家
        moe_top_k=moe_c.get("top_k", 1),  # 默认每个 token 激活 1 个专家
        moe_load_balance_weight=moe_c.get("load_balance_weight", 0.01),
        moe_router_z_loss_weight=moe_c.get("router_z_loss_weight", 0.0),
        pad_token_id=tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else 0,
        bos_token_id=tokenizer.bos_token_id
        if tokenizer.bos_token_id is not None
        else 1,
        eos_token_id=tokenizer.eos_token_id
        if tokenizer.eos_token_id is not None
        else 2,
    )


def init_tokenizer(cfg: dict) -> PreTrainedTokenizerBase:
    """初始化分词器"""
    tp = cfg["dataset"]["tokenizer_path"]
    return AutoTokenizer.from_pretrained(tp)


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    project_root = Path(__file__).resolve().parent
    cfg = load_config(str(config_path))

    # 将 pretrain 段合并到顶层，使 cfg["training"] / cfg["dataset"] / cfg["paths"] 可用
    cfg = merge_pretrain_sections(cfg)
    cfg = resolve_config_paths(cfg, project_root)
    validate_training_config(cfg)
    set_seed(args.seed)
    # 命令行参数覆盖 YAML 配置
    if args.max_steps is not None:
        cfg["training"]["max_steps"] = args.max_steps

    if torch.cuda.is_available():
        print(f"设备: CUDA - {torch.cuda.get_device_name(0)}")
    else:
        print("设备: CPU")

    print("正在加载分词器...")
    tok = init_tokenizer(cfg)
    print(f"分词器加载完成, 词表大小: {len(tok)}")

    print("正在初始化模型...")
    model = TarsLMModel(build_model_config(cfg, tok))
    total_params, size_str = model.count_parameters()
    print(f"模型初始化完成, 参数量: {total_params:,} ({size_str})")

    print("正在加载数据...")
    td, ed = load_train_eval_data(
        cfg,
        tok,
        args.train_data_path,
        seed=args.seed,
    )
    print("数据加载完成")

    # 创建处理数据整理器，设置 mlm=False 目的是禁用掩码语言建模，切换为因果语言建模
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    tc, hc, pc = cfg["training"], cfg["hardware"], cfg["paths"]

    # 配置混合精度，fp16、bf16二选一
    # fp16：半精度浮点（速度快，需 GPU 支持）
    # bf16：脑浮点（数值范围与 fp32 相同，A100/H100 推荐）
    fp16 = tc["use_mixed_precision"] and tc["precision"] == "fp16"
    bf16 = tc["use_mixed_precision"] and tc["precision"] == "bf16"

    # 评估策略 eval_strategy："steps"=按照训练步数（steps）进行评估；"epoch"=按照训练轮数（epoch）进行评估；"no"=不进行任何评估
    es = "steps" if ed is not None else "no"

    # 配置 HF TrainingArguments（所有训练超参在此集中管理）
    ta = TrainingArguments(
        output_dir=pc["checkpoint_dir"],  # checkpoint 保存目录
        per_device_train_batch_size=tc["batch_size"],  # 每张 GPU 的训练批次大小
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],  # 梯度累积步数（模拟更大 batch）
        learning_rate=tc["learning_rate"],  # 初始学习率
        warmup_steps=tc["warmup_steps"],  # 学习率预热步数（从 0 线性升至 learning_rate）
        max_steps=tc["max_steps"],  # 最大训练步数（超过则停止）
        lr_scheduler_type=tc["lr_scheduler"],  # 学习率调度器：cosine / linear / constant
        weight_decay=tc["weight_decay"],  # AdamW 权重衰减（L2 正则化系数）
        adam_beta1=tc["adam_beta1"],  # Adam 一阶矩衰减系数（默认 0.9）
        adam_beta2=tc["adam_beta2"],  # Adam 二阶矩衰减系数（默认 0.999，LLM 常用 0.95）
        adam_epsilon=tc["adam_epsilon"],  # Adam 数值稳定项（防除零）
        max_grad_norm=tc["max_grad_norm"],  # 梯度裁剪最大范数（防梯度爆炸）
        fp16=fp16,  # 启用 fp16 混合精度
        bf16=bf16,  # 启用 bf16 混合精度
        gradient_checkpointing=hc.get("enable_gradient_checkpointing", False),  # 梯度检查点（省显存）
        logging_steps=tc["log_interval"],  # 每隔多少步打印一次训练指标
        save_steps=tc["save_interval"],  # 每隔多少步保存一次 checkpoint
        eval_steps=tc["eval_interval"],  # 每隔多少步在验证集上评估
        eval_strategy=es,  # 评估策略（steps / no）
        save_total_limit=tc["keep_checkpoint_max"],  # 最多保留的 checkpoint 数量（旧的自删）
        report_to="tensorboard",  # 日志上报目标（TensorBoard / wandb）
        deepspeed=resolve_deepspeed_config_path(cfg, project_root),  # 可选 DeepSpeed 配置，缺省保持 DDP
        dataloader_num_workers=cfg["dataset"].get("num_workers", 0),  # 数据加载子进程数（0=主进程加载）
        dataloader_pin_memory=True,  # 锁页内存：加速数据从 CPU 内存向 GPU 显存的传输速度
        remove_unused_columns=True,  # 在训练前自动从数据集中移除模型前向传播（Forward）不需要的多余列
        load_best_model_at_end=False,  # 训练结束时不自动加载最优模型
    )

    # 构建 Trainer 并启动训练
    trainer = Trainer(
        model=model,
        args=ta,
        train_dataset=td,
        eval_dataset=ed,
        data_collator=collator,
    )

    resume = None
    if args.resume is not None:
        # resume=True时，框架会自动在指定的输出目录中寻找最新的 Checkpoint 文件并从中恢复训练。
        resume = True if args.resume.lower() in ("latest", "true") else args.resume

    print("开始预训练...")
    trainer.train(resume_from_checkpoint=resume)
    checkpoint_dir = pc["checkpoint_dir"]
    tok.save_pretrained(checkpoint_dir)
    print(f"预训练完成，checkpoint 已保存至 {checkpoint_dir}")


if __name__ == "__main__":
    main()
