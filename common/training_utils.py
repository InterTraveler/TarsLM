"""训练配置辅助函数。

预训练入口和对话微调入口共用本模块，用来处理三件容易混淆的事情：

1. 把 YAML 中嵌套的 ``pretrain`` 配置提升到顶层；
2. 在预训练默认值之上叠加 ``finetune`` 覆盖值；
3. 判断是否启用 DeepSpeed，并把相对路径统一解析为项目内绝对路径。

这些函数只处理配置字典和路径，不创建模型、不加载数据，因此可以在普通
CPU 环境下安全测试，不会触发服务器规模模型的显存溢出。
"""

from pathlib import Path
from typing import Any

import torch

TRAINING_REQUIRED_FIELDS = (
    "batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "warmup_steps",
    "max_steps",
    "lr_scheduler",
    "max_grad_norm",
    "weight_decay",
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
    "log_interval",
    "save_interval",
    "eval_interval",
    "keep_checkpoint_max",
    "use_mixed_precision",
    "precision",
)

MODEL_REQUIRED_FIELDS = (
    "vocab_size",
    "hidden_size",
    "num_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "max_seq_len",
)


def resolve_device() -> torch.device:
    """返回当前环境可用的推理/训练设备。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _require_fields(cfg: dict[str, Any], section: str, fields) -> None:
    """检查配置段中的必填字段。"""
    section_cfg = cfg.get(section) or {}
    missing = [field for field in fields if field not in section_cfg]
    if missing:
        raise ValueError(
            f"配置文件缺少 {section} 字段: {', '.join(missing)}"
        )


def _require_positive_int(cfg: dict[str, Any], section: str, field: str) -> None:
    """检查正整数配置。"""
    value = cfg[section][field]
    if type(value) is not int or value <= 0:
        raise TypeError(f"{section}.{field} 必须为正整数")


def validate_training_config(cfg: dict[str, Any]) -> None:
    """校验预训练与微调入口所需的配置结构。"""
    required_sections = ("model", "moe", "hardware", "training", "dataset", "paths")
    missing = [section for section in required_sections if section not in cfg]
    if missing:
        raise ValueError(f"配置文件缺少顶层字段: {', '.join(missing)}")

    _require_fields(cfg, "model", MODEL_REQUIRED_FIELDS)
    for field in (
        "vocab_size",
        "hidden_size",
        "num_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "max_seq_len",
    ):
        _require_positive_int(cfg, "model", field)

    _require_fields(cfg, "training", TRAINING_REQUIRED_FIELDS)
    training = cfg["training"]
    if not isinstance(training["use_mixed_precision"], bool):
        raise TypeError("training.use_mixed_precision 必须为布尔值")
    if training["precision"] not in {"fp16", "bf16"}:
        raise ValueError("training.precision 仅支持 fp16 或 bf16")

    dataset = cfg["dataset"]
    if not dataset.get("data_format"):
        raise ValueError("dataset.data_format 不能为空")
    if not dataset.get("tokenizer_path"):
        raise ValueError("dataset.tokenizer_path 不能为空")
    if not dataset.get("train_data_path"):
        raise ValueError("dataset.train_data_path 不能为空")

    paths = cfg["paths"]
    if not paths.get("checkpoint_dir"):
        raise ValueError("paths.checkpoint_dir 不能为空")


def merge_pretrain_sections(cfg: dict[str, Any]) -> dict[str, Any]:
    """把 ``pretrain`` 中的三个子段复制到顶层。

    入口脚本习惯通过 ``cfg["training"]``、``cfg["dataset"]`` 和
    ``cfg["paths"]`` 访问配置，而 YAML 中这些字段位于 ``pretrain`` 下。
    这里统一提升到顶层，让预训练和微调代码可以复用同一套读取方式。

    函数会返回一个新字典，不修改调用方传入的 ``cfg``。
    """
    merged = dict(cfg)
    pretrain = merged.get("pretrain") or {}

    # 只复制实际存在的段，避免把 None 或空配置写入顶层。
    for section in ("training", "dataset", "paths"):
        if section in pretrain:
            merged[section] = dict(pretrain[section])

    return merged


def apply_finetune_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    """合并预训练默认值和微调覆盖值。

    微调配置通常只需要覆盖少数字段，例如 ``max_steps``、``learning_rate``
    或 ``train_data_path``。本函数先把预训练段提升到顶层，再逐项用
    ``finetune`` 的同名字段覆盖，未覆盖的字段继续继承预训练值。

    这样既能避免在两个 YAML 中重复维护完整训练参数，又能保证微调逻辑
    读到的是最终生效的完整配置。
    """
    merged = merge_pretrain_sections(cfg)
    finetune = merged.get("finetune") or {}

    for section in ("training", "dataset", "paths"):
        if section not in finetune:
            continue

        # 先复制已有默认值，再更新微调字段，避免破坏继承关系。
        section_cfg = dict(merged.get(section) or {})
        section_cfg.update(finetune[section])
        merged[section] = section_cfg

    return merged


def resolve_deepspeed_config_path(
    cfg: dict[str, Any],
    project_root: Path | None = None,
) -> str | None:
    """解析可选的 DeepSpeed 配置文件路径。

    设计目标是完全向后兼容：

    - 字段不存在、为 ``null``、为 ``false`` 或空字符串时，返回 ``None``，
      继续使用原来的 DDP 行为；
    - 字段为字符串时，按项目根目录解析相对路径，避免从其他工作目录启动时
      找不到配置；绝对路径则原样返回；
    - 字段为其他类型时直接抛出 ``ValueError``。

    ``project_root`` 用于单元测试注入临时目录；生产代码应传入项目根目录。
    """
    root = (
        Path(project_root)
        if project_root is not None
        else Path(__file__).resolve().parent.parent
    )
    hardware = cfg.get("hardware") or {}
    value = hardware.get("deepspeed_config")

    if value is None or value is False or value == "":
        return None
    if not isinstance(value, str):
        raise TypeError(
            "hardware.deepspeed_config 必须是字符串路径，"
            "也可以省略、设为 null 或 false 以保持 DDP"
        )

    path = Path(value)
    if path.is_absolute():
        return str(path)

    return str((root / path).resolve())


def resolve_config_paths(cfg: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """把 YAML 中的相对路径解析到项目根目录。

    Hugging Face Trainer 的 ``output_dir``、数据路径和分词器路径都不应依赖
    启动脚本时的当前工作目录，因此在这里统一解析一次。
    """
    base = Path(project_root).resolve()

    def resolve(path: Any) -> Any:
        if not isinstance(path, str) or not path:
            return path
        parsed = Path(path)
        return str(parsed) if parsed.is_absolute() else str((base / parsed).resolve())

    dataset = cfg.get("dataset") or {}
    for key in ("train_data_path", "tokenizer_path"):
        if key in dataset:
            dataset[key] = resolve(dataset[key])

    paths = cfg.get("paths") or {}
    if "checkpoint_dir" in paths:
        paths["checkpoint_dir"] = resolve(paths["checkpoint_dir"])

    finetune = cfg.get("finetune") or {}
    if "pretrained_checkpoint" in finetune:
        finetune["pretrained_checkpoint"] = resolve(
            finetune["pretrained_checkpoint"]
        )
    return cfg
