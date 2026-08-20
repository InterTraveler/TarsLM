# TarsLM

[English](README.md) | **中文**

TarsLM 是一个面向消费级显卡的小型 LLM 预训练项目。它从零实现了 Decoder-Only Transformer，并打通了从 Hugging Face fast tokenizer 训练、语料加载、预训练、对话微调到交互式聊天的完整流程；同时与 Hugging Face 生态兼容，可以直接使用 `Trainer` 训练、`generate()` 推理、`AutoTokenizer` 加载分词器。

代码按模块拆分、中文注释详细、配置开箱即用，既适合在 8GB 显存级别的单卡上训练一个小模型，也适合作为学习 LLM 内部实现的参考工程。

## 特性

- 完整训练链路：分词器训练 → JSON 语料加载 → HF Trainer 预训练 → 对话微调 → checkpoint 保存/断点续训 → 采样生成
- 交互式对话：支持 `tokenizer.apply_chat_template()` 格式化与 `StoppingCriteria` 停止控制
- Llama 风格架构：RMSNorm + RoPE + SwiGLU + Pre-Norm 残差 + KV Cache + 权重绑定
- MHA / GQA 可切换：`num_key_value_heads < num_attention_heads` 时自动使用 GQA，节省 KV Cache 显存
- 可选 MoE：标准 SwiGLU FFN 与 MoE（top-k 稀疏路由）一键切换
- 显存优化：梯度检查点、fp16/bf16 混合精度、梯度累积
- Flash Attention 兼容：通过 PyTorch SDPA 在支持的 GPU 上自动选择 FlashAttention 后端
- 分布式训练可选：`server.yaml` 可启用 DeepSpeed ZeRO-2；其他配置在不使用 `torchrun` 等多进程启动器时按单进程运行
- Hugging Face 兼容：继承 `PreTrainedModel` + `GenerationMixin`，分词器统一使用 `AutoTokenizer.from_pretrained()` 加载
- 三套开箱配置：快速调试 / 消费级单卡 / 服务器集群
- 分层单元测试：从基础模块到完整模型逐层覆盖，固定随机种子保证可复现

## 模型架构

| 模块 | 文件 | 说明 |
| --- | --- | --- |
| 配置 | `model/config.py` | `TarsLMConfig`，继承 `PretrainedConfig` |
| 位置编码 | `model/rope.py` | RoPE 旋转位置编码，预计算 cos/sin 频率表 |
| 归一化 | `model/norm.py` | RMSNorm |
| 注意力 | `model/attention.py` | MHA / GQA / PyTorch SDPA / KV Cache |
| 前馈网络 | `model/ffn.py` | SwiGLU FFN 与 `TarsLMMoE` |
| 解码层 | `model/decoder_layer.py` | Pre-Norm + 残差，梯度检查点包装 |
| 顶层模型 | `model/model.py` | `TarsLMModel`，CLM 损失与 `generate()` |

`TarsLMConfig` 主要参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `vocab_size` | 8000 | 词表大小，需与分词器一致 |
| `hidden_size` | 256 | 隐藏层维度 |
| `num_layers` | 6 | Transformer 层数 |
| `num_attention_heads` | 8 | Q 头数 |
| `num_key_value_heads` | 8 | KV 头数，小于 Q 头数时启用 GQA |
| `intermediate_size` | 1024 | FFN 中间层维度 |
| `max_seq_len` | 1024 | 最大序列长度 |
| `use_moe` | True | 是否用 MoE 替换标准 FFN |
| `num_experts` / `moe_top_k` | 2 / 1 | 专家总数 / 每个 token 激活的专家数 |
| `rope_theta` | 10000.0 | RoPE 旋转基频 |

完整参数说明见 `model/config.py`。

## 快速开始

### 1. 环境准备

Python 3.10+，推荐 3.11；Windows / Linux 均可。

> 本项目按源码方式运行，不提供 pip 命令行入口。请克隆后在项目根目录执行下面的
> `python ...` 命令。

```bash
pip install -r requirements.txt
```

开发与测试依赖：

```bash
pip install -r requirements-dev.txt
```

PyTorch 需要与本地 CUDA 版本匹配，例如 CUDA 12.4：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

运行 `config/server.yaml` 时还需要安装可选的 DeepSpeed：

```bash
pip install deepspeed
```

### 2. 准备数据

预训练语料放在 `data/train_data/multi/`，对话语料放在 `data/train_data/chat/`，分词器产物保存在 `data/tokenizer/`。支持 `.txt`、`.json`、`.jsonl`、`.parquet` 四种格式，通过 `data_format` 指定后会递归加载。文本统一取自 `text` 字段，parquet 额外支持 `content` 列。

**预训练数据**（`data/train_data/multi/train_data.json`）：

```json
[
  {"text": "第一句话。"},
  {"text": "第二句话。"}
]
```

**对话数据**（`data/train_data/chat/chat_data.json`）：

```json
[
  {"text": "<|user|>你好，你是谁？<|assistant|>你好！我是 TarsLM。<|end|>"},
  {"text": "<|user|>再见<|assistant|>再见！祝你愉快！<|end|>"}
]
```

> 对话格式使用 `<|user|>` / `<|assistant|>` / `<|end|>` 三个特殊 token 标记角色和轮次边界。`train_tokenizer.py` 会在训练分词器时预留这些 token；`chat_finetune.py` 会校验并兼容旧分词器完成注册，同时设置 `chat_template`，推理时通过 `tokenizer.apply_chat_template()` 格式化对话 prompt。

### 3. 训练分词器

新 clone 后可以直接用 bootstrap 脚本一键完成默认分词器、预训练和对话微调：

```bash
python bootstrap.py
```

默认会依次执行，训练分词器时同时使用预训练与微调语料：

1. 训练 `config/default.yaml` 所需的分词器；
2. 运行默认预训练配置；
3. 在预训练 checkpoint 上运行默认对话微调配置。

如果只需要执行部分阶段，可以使用：

```bash
python bootstrap.py --skip_pretrain --skip_finetune
```

也可以手动训练分词器：

```bash
python train_tokenizer.py --corpus_paths ./data/train_data/multi/ ./data/train_data/chat/ --data_format json --vocab_size 1000 --output ./data/tokenizer/
```

Python API 中对应参数为 `corpus_paths`：

```python
from train_tokenizer import train_tokenizer

train_tokenizer(
    corpus_paths=[
        "./data/train_data/multi/",
        "./data/train_data/chat/",
    ],
    output_dir="./data/tokenizer/",
    vocab_size=1000,
    data_format="json",
)
```

`train_tokenizer.py` 参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--corpus` | `./data/train_data/multi/` | 语料目录或单个文件路径；传目录时递归加载 |
| `--corpus_paths` | 无 | 多个语料目录或单文件路径；传入后优先于 `--corpus` |
| `--data_format` | 必填 | 语料格式：`txt` / `json` / `jsonl` / `parquet` |
| `--output` | `./data/tokenizer/` | 分词器保存目录 |
| `--vocab_size` | `8000` | 目标词表大小 |
| `--model_type` | `unigram` | 分词算法：`bpe` / `unigram` |
| `--text_column` | 自动识别 | parquet 文本列名；默认自动识别 `text` / `content` |

### 4. 预训练

先用冒烟配置验证环境（极小参数规模、少量步数）：

```bash
python main_pretrain.py --config config/default.yaml
```

消费级单卡配置（较小参数规模、8GB 显存、fp16）：

> 启动前请先把 `config/consumer.yaml` 中的
> `pretrain.dataset.train_data_path` 和 `finetune.dataset.train_data_path`
> 修改为实际语料目录；仓库中的 `/data/...` 只是占位路径。

```bash
python main_pretrain.py --config config/consumer.yaml
```

服务器集群配置使用 DeepSpeed ZeRO-2，需要先安装可选的 `deepspeed` 依赖，并把
`config/server.yaml` 中的预训练和微调语料路径改成实际路径：

```bash
torchrun --nproc_per_node=8 main_pretrain.py --config config/server.yaml
```

> `config/server.yaml` 面向 8×A100/H100，约 6.7B 总参数，普通本地环境会显存溢出，
> 不建议在本地开发机上启动该配置。该 DeepSpeed ZeRO-2 路径目前尚未完成端到端实测。

### 5. 断点续训

```bash
python main_pretrain.py --config config/consumer.yaml --resume latest
```

### 6. 文本生成测试

加载最新 checkpoint 并生成文本：

```bash
python eval.py
```

代码中直接使用：

```python
from transformers import AutoTokenizer

from model.model import TarsLMModel

tokenizer = AutoTokenizer.from_pretrained("./data/tokenizer/")
model = TarsLMModel.from_pretrained("./checkpoints/checkpoint-6000")

inputs = tokenizer("人工智能", return_tensors="pt")
output_ids = model.generate(
    **inputs,
    max_new_tokens=64,
    do_sample=True,
    top_p=0.95,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

### 7. 对话微调

在预训练 checkpoint 基础上，用对话数据继续训练（全量微调）：

```bash
# 微调 debug 配置
python chat_finetune.py --config config/default.yaml

# 微调消费级单卡配置
# 同样需要先修改 consumer.yaml 中的 /data/chat_corpus/ 为实际路径
python chat_finetune.py --config config/consumer.yaml

# 微调服务器集群配置
# server.yaml 使用 /data/chat_corpus/ 占位路径，也需要先改成实际路径
python chat_finetune.py --config config/server.yaml
```

`chat_finetune.py` 参数：

| 参数 | 说明 |
| --- | --- |
| `--config` | 配置文件路径，默认 `config/default.yaml` |
| `--max_steps` | 覆盖最大训练步数 |
| `--resume` | `latest`/`true` 自动恢复最新微调 checkpoint，或指定具体路径 |
| `--seed` | 随机种子，默认 `42` |

### 8. 交互式对话

加载微调后的模型，启动终端对话：

```bash
python chat.py
```

`chat.py` 参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--checkpoint_dir` | `./checkpoints_chat` | 微调输出目录（自动选择最新 checkpoint-n） |
| `--tokenizer` | 自动推断 | 分词器目录；默认从 checkpoint 目录或上级目录推断 |
| `--max_new_tokens` | `128` | 每次回复最多生成的 token 数 |
| `--temperature` | `0.8` | 采样温度 |
| `--top_p` | `0.95` | Nucleus 采样阈值 |

每次对话为全新会话，不携带历史记录。输入 `exit` 或 `quit` 退出。

### 9. 查看训练曲线

```bash
tensorboard --logdir ./checkpoints
```

## 命令行参数

`main_pretrain.py` 支持通过命令行覆盖 YAML 配置：

| 参数 | 说明 |
| --- | --- |
| `--config` | 配置文件路径，默认 `config/default.yaml` |
| `--max_steps` | 覆盖最大训练步数 |
| `--resume` | `latest`/`true` 自动恢复最新 checkpoint，或指定具体路径 |
| `--train_data_path` | 覆盖语料目录，递归加载 `dataset.data_format` 指定格式的文件 |
| `--seed` | 随机种子，默认 `42` |

验证集按文件名 `eval_*` / `valid_*` 前缀自动识别；不存在验证文件时跳过评估。

## 配置文件

项目提供三套开箱配置，预训练与微调共用：

| 文件 | 目标硬件 | 模型规模 | 混合精度 | 说明 |
| --- | --- | --- | --- | --- |
| `config/default.yaml` | 任意 CPU/GPU | 极小参数 | 关闭 | 快速冒烟测试，少量步数验证链路 |
| `config/consumer.yaml` | 8GB 单卡 | 较小参数 | fp16 | GQA + 梯度检查点，适合个人日常训练 |
| `config/server.yaml` | 8 x A100/H100 | 较大参数 | bf16 | MoE 8 专家 + PyTorch SDPA + DeepSpeed ZeRO-2 大规模预训练；尚未完成端到端实测 |

> 模型规模按当前代码实际实例化统计，与部分 YAML 注释中的估算值可能存在差异。

每个 YAML 按 `model`、`moe`、`hardware`、`pretrain`、`finetune` 五个顶层区块组织；预训练和微调的训练超参分别位于 `pretrain` 和 `finetune` 下的 `training`、`dataset`、`paths` 子段。`finetune` 段仅在微调时读取，并只覆盖预训练配置中的同名字段。

服务器配置额外使用 `config/deepspeed_zero2.json` 启用 DeepSpeed ZeRO-2，但该分布式路径目前尚未完成端到端实测。`hardware.deepspeed_config` 不存在、为 `null` 或 `false` 时，不会传入 DeepSpeed 配置；是否分布式训练由启动器决定。直接运行 `python main_pretrain.py` 为单进程，使用 `torchrun --nproc_per_node=...` 才启用多进程分布式训练。

## 项目结构

```text
TarsLM/
├── config/                     # 训练配置（预训练与微调共用）
│   ├── default.yaml            # 预训练冒烟配置
│   ├── consumer.yaml           # 消费级单卡预训练
│   ├── server.yaml             # 服务器集群预训练
│   └── deepspeed_zero2.json    # 服务器 ZeRO-2 分布式配置
├── common/                     # 公共工具
│   ├── __init__.py
│   ├── checkpoint.py           # checkpoint 自动发现
│   ├── data_io.py              # 公共文件解析工具
│   └── training_utils.py       # 训练配置合并与 DeepSpeed 路径解析
├── data/                       # 数据与分词器
│   ├── __init__.py
│   ├── train_data/             # 训练语料
│   │   ├── chat/               # 对话微调语料
│   │   │   ├── chat_data.json
│   │   │   └── eval_chat_data.json
│   │   └── multi/              # 预训练语料
│   │       ├── eval_data.json
│   │       └── train_data.json
│   ├── tokenizer/              # 分词器产物，包含 tokenizer.json（gitignore）
│   └── data_loader.py          # 训练/验证数据加载
├── model/                      # 模型核心代码
│   ├── config.py
│   ├── rope.py
│   ├── norm.py
│   ├── attention.py
│   ├── ffn.py
│   ├── decoder_layer.py
│   ├── model.py
│   └── __init__.py
├── tests/                      # 分层单元测试
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_data_loading.py
│   ├── test_training_config.py
│   ├── test_bootstrap.py
│   ├── test_checkpoint.py
│   ├── test_config_validation.py
│   ├── test_tokenizer_integration.py
│   ├── test_l1_rope.py
│   ├── test_l2_rmsnorm.py
│   ├── test_l3_rotary.py
│   ├── test_l4_attention.py
│   ├── test_l5_ffn.py
│   ├── test_l6_decoder.py
│   ├── test_l7_model.py
│   └── utils.py
├── checkpoints/                # 预训练产物（gitignore）
├── checkpoints_chat/           # 对话微调产物（gitignore）
├── main_pretrain.py            # 预训练入口
├── bootstrap.py                # 一键生成分词器并完成预训练、微调
├── chat_finetune.py            # 对话微调入口
├── chat.py                     # 交互式对话
├── eval.py                     # 文本生成测试入口
├── train_tokenizer.py          # 分词器训练脚本
├── pyproject.toml              # 项目元数据与 lint 配置
├── requirements.txt
├── requirements-dev.txt
└── .gitignore
```

## 测试

```bash
pytest tests/ -v
```

测试从 RoPE、RMSNorm 等基础模块到完整模型端到端、过拟合、权重绑定、KV Cache、HF 接口均有覆盖。测试自动选择 CPU / CUDA，并通过全局与逐用例随机种子保证可复现。

## 致谢

- [Llama (Meta)](https://arxiv.org/abs/2302.13971)：RMSNorm、RoPE、SwiGLU、GQA 等架构设计
- [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers)：fast tokenizer 训练与序列化
- [Hugging Face Transformers](https://github.com/huggingface/transformers)：模型接口与 Trainer 训练框架
- [PyTorch](https://pytorch.org/)：核心计算框架

## License

MIT. 详见 [LICENSE](LICENSE)。
