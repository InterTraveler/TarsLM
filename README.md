# TarsLM

**English** | [中文](README_zh.md)

TarsLM is a small LLM pretraining project aimed at consumer GPUs. It implements a decoder-only Transformer from scratch and covers the complete workflow, from training a Hugging Face fast tokenizer and loading corpora to pretraining, chat fine-tuning, and interactive chat. It is also compatible with the Hugging Face ecosystem, so you can train with `Trainer`, generate with `generate()`, and load the tokenizer with `AutoTokenizer`.

The code is split into focused modules, documented in detail, and configured to run out of the box. It is suitable both for training a small model on a single 8 GB GPU and as a reference project for learning LLM internals.

## Features

- Complete training pipeline: tokenizer training -> JSON corpus loading -> HF Trainer pretraining -> chat fine-tuning -> checkpoint saving/resume -> sampled generation
- Interactive chat: `tokenizer.apply_chat_template()` formatting and `StoppingCriteria`-based stopping
- Llama-style architecture: RMSNorm + RoPE + SwiGLU + pre-norm residual connections + KV cache + tied weights
- Switchable MHA/GQA: GQA is used automatically when `num_key_value_heads < num_attention_heads`, reducing KV cache memory
- Optional MoE: switch between the standard SwiGLU FFN and MoE top-k sparse routing with one configuration
- Memory optimizations: gradient checkpointing, fp16/bf16 mixed precision, gradient accumulation
- Flash Attention compatibility: PyTorch SDPA automatically selects the FlashAttention backend on supported GPUs
- Optional distributed training: `server.yaml` can enable DeepSpeed ZeRO-2; other configs run single-process unless launched with a multi-process tool such as `torchrun`
- Hugging Face compatibility: models inherit `PreTrainedModel` + `GenerationMixin`, and tokenizers are loaded with `AutoTokenizer.from_pretrained()`
- Three out-of-the-box configs: quick debugging, consumer single-GPU, and server cluster
- Layered unit tests: coverage from basic modules to the complete model, with fixed random seeds for reproducibility

## Model Architecture

| Module | File | Description |
| --- | --- | --- |
| Config | `model/config.py` | `TarsLMConfig`, inheriting `PretrainedConfig` |
| Position encoding | `model/rope.py` | RoPE rotary position encoding with precomputed cos/sin frequency tables |
| Normalization | `model/norm.py` | RMSNorm |
| Attention | `model/attention.py` | MHA / GQA / PyTorch SDPA / KV cache |
| Feed-forward network | `model/ffn.py` | SwiGLU FFN and `TarsLMMoE` |
| Decoder layer | `model/decoder_layer.py` | Pre-norm + residual, wrapped with gradient checkpointing |
| Top-level model | `model/model.py` | `TarsLMModel`, CLM loss, and `generate()` |

Key `TarsLMConfig` parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `vocab_size` | 8000 | Vocabulary size; must match the tokenizer |
| `hidden_size` | 256 | Hidden dimension |
| `num_layers` | 6 | Number of Transformer layers |
| `num_attention_heads` | 8 | Number of query heads |
| `num_key_value_heads` | 8 | Number of KV heads; GQA is enabled when this is smaller than the query head count |
| `intermediate_size` | 1024 | FFN intermediate dimension |
| `max_seq_len` | 1024 | Maximum sequence length |
| `use_moe` | True | Whether to replace the standard FFN with MoE |
| `num_experts` / `moe_top_k` | 2 / 1 | Total experts / experts activated per token |
| `rope_theta` | 10000.0 | RoPE rotary base frequency |

See `model/config.py` for the complete parameter reference.

## Quick Start

### 1. Environment Setup

Python 3.10+ is required; Python 3.11 is recommended. Windows and Linux are both supported.

> This project runs from source and does not provide a pip CLI entry point. Clone the repository, then run the `python ...` commands from the project root.

```bash
pip install -r requirements.txt
```

Development and test dependencies:

```bash
pip install -r requirements-dev.txt
```

PyTorch should match your local CUDA version, for example CUDA 12.4:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Running `config/server.yaml` also requires the optional DeepSpeed dependency:

```bash
pip install deepspeed
```

### 2. Prepare the Data

Pretraining corpora go in `data/train_data/multi/`, chat corpora in `data/train_data/chat/`, and tokenizer artifacts in `data/tokenizer/`. The loader supports `.txt`, `.json`, `.jsonl`, and `.parquet`; set `data_format` to load files recursively. Text is read from the `text` field, and parquet additionally supports a `content` column.

**Pretraining data** (`data/train_data/multi/train_data.json`):

```json
[
  {"text": "First sentence."},
  {"text": "Second sentence."}
]
```

**Chat data** (`data/train_data/chat/chat_data.json`):

```json
[
  {"text": "<|user|>Hello, who are you?<|assistant|>Hello! I am TarsLM.<|end|>"},
  {"text": "<|user|>Goodbye<|assistant|>Goodbye! Have a great day!<|end|>"}
]
```

> The chat format uses three special tokens, `<|user|>`, `<|assistant|>`, and `<|end|>`, to mark roles and turn boundaries. These tokens are reserved when `train_tokenizer.py` trains the tokenizer; `chat_finetune.py` validates and registers them for compatibility with older tokenizers, then sets `chat_template`. At inference time, prompts are formatted with `tokenizer.apply_chat_template()`.

### 3. Train the Tokenizer

After a fresh clone, `bootstrap.py` can train the default tokenizer, pretrain, and fine-tune for chat in one command:

```bash
python bootstrap.py
```

By default it runs the following steps, using both pretraining and fine-tuning corpora for tokenizer training:

1. Train the tokenizer required by `config/default.yaml`.
2. Run the default pretraining config.
3. Run the default chat fine-tuning config on the pretraining checkpoint.

To run only some stages:

```bash
python bootstrap.py --skip_pretrain --skip_finetune
```

You can also train the tokenizer manually:

```bash
python train_tokenizer.py --corpus_paths ./data/train_data/multi/ ./data/train_data/chat/ --data_format json --vocab_size 1000 --output ./data/tokenizer/
```

The equivalent Python API parameter is `corpus_paths`:

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

`train_tokenizer.py` arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `--corpus` | `./data/train_data/multi/` | Corpus directory or single file path; directories are loaded recursively |
| `--corpus_paths` | None | Multiple corpus directories or single file paths; takes precedence over `--corpus` |
| `--data_format` | Required | Corpus format: `txt` / `json` / `jsonl` / `parquet` |
| `--output` | `./data/tokenizer/` | Tokenizer output directory |
| `--vocab_size` | `8000` | Target vocabulary size |
| `--model_type` | `unigram` | Tokenization algorithm: `bpe` / `unigram` |
| `--text_column` | Auto-detected | Parquet text column name; auto-detects `text` / `content` |

### 4. Pretraining

Validate the environment with the smoke-test config first:

```bash
python main_pretrain.py --config config/default.yaml
```

Consumer single-GPU config:

> Before starting, update `pretrain.dataset.train_data_path` and `finetune.dataset.train_data_path` in `config/consumer.yaml` to your actual corpus directories; the `/data/...` paths in the repository are placeholders.

```bash
python main_pretrain.py --config config/consumer.yaml
```

The server-cluster config uses DeepSpeed ZeRO-2. Install the optional `deepspeed` dependency and update the pretraining and fine-tuning corpus paths in `config/server.yaml` first:

```bash
torchrun --nproc_per_node=8 main_pretrain.py --config config/server.yaml
```

> `config/server.yaml` targets 8 x A100/H100 and has roughly 6.7B total parameters. Ordinary local environments will run out of memory, so this config is not recommended on a local development machine. The DeepSpeed ZeRO-2 path has not been verified end-to-end yet.

### 5. Resume Training

```bash
python main_pretrain.py --config config/consumer.yaml --resume latest
```

### 6. Text Generation Test

Load the latest checkpoint and generate text:

```bash
python eval.py
```

Direct use in Python:

```python
from transformers import AutoTokenizer

from model.model import TarsLMModel

tokenizer = AutoTokenizer.from_pretrained("./data/tokenizer/")
model = TarsLMModel.from_pretrained("./checkpoints/checkpoint-6000")

inputs = tokenizer("artificial intelligence", return_tensors="pt")
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

### 7. Chat Fine-Tuning

Continue training on chat data from a pretraining checkpoint:

```bash
# Fine-tune the debug config
python chat_finetune.py --config config/default.yaml

# Fine-tune the consumer single-GPU config
# Update /data/chat_corpus/ in consumer.yaml to your actual path first
python chat_finetune.py --config config/consumer.yaml

# Fine-tune the server cluster config
# server.yaml uses /data/chat_corpus/ as a placeholder; update it first
python chat_finetune.py --config config/server.yaml
```

`chat_finetune.py` arguments:

| Argument | Description |
| --- | --- |
| `--config` | Config file path; defaults to `config/default.yaml` |
| `--max_steps` | Override the maximum number of training steps |
| `--resume` | `latest`/`true` to resume from the latest fine-tuning checkpoint, or a specific path |
| `--seed` | Random seed; defaults to `42` |

### 8. Interactive Chat

Load a fine-tuned model and start a terminal conversation:

```bash
python chat.py
```

`chat.py` arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `--checkpoint_dir` | `./checkpoints_chat` | Fine-tuning output directory; the latest `checkpoint-n` is selected automatically |
| `--tokenizer` | Auto-inferred | Tokenizer directory; inferred from the checkpoint directory or parent directory by default |
| `--max_new_tokens` | `128` | Maximum generated tokens per reply |
| `--temperature` | `0.8` | Sampling temperature |
| `--top_p` | `0.95` | Nucleus sampling threshold |

Every chat session starts fresh and does not carry history. Enter `exit` or `quit` to stop.

### 9. View Training Curves

```bash
tensorboard --logdir ./checkpoints
```

## Command-Line Arguments

`main_pretrain.py` supports overriding YAML config values from the command line:

| Argument | Description |
| --- | --- |
| `--config` | Config file path; defaults to `config/default.yaml` |
| `--max_steps` | Override the maximum number of training steps |
| `--resume` | `latest`/`true` to resume from the latest checkpoint, or a specific path |
| `--train_data_path` | Override the corpus directory; recursively loads files in the format specified by `dataset.data_format` |
| `--seed` | Random seed; defaults to `42` |

Validation files are auto-detected by the `eval_*` / `valid_*` filename prefixes. Evaluation is skipped when no validation files exist.

## Configuration Files

The project provides three out-of-the-box configs, shared by pretraining and fine-tuning:

| File | Target Hardware | Model Size | Mixed Precision | Description |
| --- | --- | --- | --- | --- |
| `config/default.yaml` | Any CPU/GPU | Very small | Off | Quick smoke test with a small number of steps |
| `config/consumer.yaml` | 8 GB single GPU | Small | fp16 | GQA + gradient checkpointing, suitable for personal daily training |
| `config/server.yaml` | 8 x A100/H100 | Larger | bf16 | MoE with 8 experts + PyTorch SDPA + DeepSpeed ZeRO-2 for large-scale pretraining; not yet verified end-to-end |

> Model sizes are measured by instantiating the current code and may differ from estimates in some YAML comments.

Each YAML file is organized into five top-level blocks: `model`, `moe`, `hardware`, `pretrain`, and `finetune`. Pretraining and fine-tuning hyperparameters live under the `training`, `dataset`, and `paths` sub-blocks of `pretrain` and `finetune` respectively. The `finetune` block is read only during fine-tuning and only overrides fields of the same name from the pretraining config.

The server config additionally uses `config/deepspeed_zero2.json` to enable DeepSpeed ZeRO-2, but this distributed path is not yet verified end-to-end. When `hardware.deepspeed_config` is missing, `null`, or `false`, no DeepSpeed config is passed; distribution is then determined by the launcher. `python main_pretrain.py` runs single-process, while `torchrun --nproc_per_node=...` enables multi-process distributed training.

## Project Structure

```text
TarsLM/
├── config/                     # Training configs shared by pretraining and fine-tuning
│   ├── default.yaml            # Pretraining smoke-test config
│   ├── consumer.yaml           # Consumer single-GPU pretraining
│   ├── server.yaml             # Server cluster pretraining
│   └── deepspeed_zero2.json    # Server ZeRO-2 distributed config
├── common/                     # Shared utilities
│   ├── __init__.py
│   ├── checkpoint.py           # Checkpoint auto-discovery
│   ├── data_io.py              # Shared file parsing helpers
│   └── training_utils.py       # Config merging and DeepSpeed path resolution
├── data/                       # Data and tokenizer
│   ├── __init__.py
│   ├── train_data/             # Training corpora
│   │   ├── chat/               # Chat fine-tuning corpora
│   │   │   ├── chat_data.json
│   │   │   └── eval_chat_data.json
│   │   └── multi/              # Pretraining corpora
│   │       ├── eval_data.json
│   │       └── train_data.json
│   ├── tokenizer/              # Tokenizer artifacts, including tokenizer.json (gitignored)
│   └── data_loader.py          # Training/validation data loading
├── model/                      # Core model code
│   ├── config.py
│   ├── rope.py
│   ├── norm.py
│   ├── attention.py
│   ├── ffn.py
│   ├── decoder_layer.py
│   ├── model.py
│   └── __init__.py
├── tests/                      # Layered unit tests
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
├── checkpoints/                # Pretraining outputs (gitignored)
├── checkpoints_chat/           # Chat fine-tuning outputs (gitignored)
├── main_pretrain.py            # Pretraining entry point
├── bootstrap.py                # One-command tokenizer, pretraining, and fine-tuning
├── chat_finetune.py            # Chat fine-tuning entry point
├── chat.py                     # Interactive chat
├── eval.py                     # Text generation test entry point
├── train_tokenizer.py          # Tokenizer training script
├── pyproject.toml              # Project metadata and lint config
├── requirements.txt
├── requirements-dev.txt
└── .gitignore
```

## Tests

```bash
pytest tests/ -v
```

The tests cover basic modules such as RoPE and RMSNorm, end-to-end model behavior, overfitting, tied weights, KV cache, and Hugging Face interfaces. Tests automatically select CPU or CUDA and use global and per-case random seeds for reproducibility.

## Acknowledgements

- [Llama (Meta)](https://arxiv.org/abs/2302.13971): RMSNorm, RoPE, SwiGLU, GQA, and other architectural designs
- [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers): fast tokenizer training and serialization
- [Hugging Face Transformers](https://github.com/huggingface/transformers): model interfaces and the Trainer training framework
- [PyTorch](https://pytorch.org/): core computation framework

## License

MIT. See [LICENSE](LICENSE).
