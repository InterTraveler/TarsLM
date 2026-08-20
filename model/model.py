"""TarsLM Decoder-Only 语言模型的顶层组装。

将 Embedding、多层 DecoderLayer、RMSNorm 与 lm_head 组合为完整模型，
并继承 PreTrainedModel 与 GenerationMixin，兼容 Hugging Face 的
保存/加载与 generate() 生成接口。
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from model.attention import TarsLMAttention  # noqa: F401 — 向后兼容重新导出
from model.config import TarsLMConfig
from model.decoder_layer import TarsLMDecoderLayer
from model.ffn import TarsLMFeedForward, TarsLMMoE  # noqa: F401 — 向后兼容重新导出
from model.norm import TarsLMRMSNorm
from model.rope import TarsLMRotaryEmbedding, _rotate_half  # noqa: F401 — 向后兼容重新导出

# =========================================================================
#  TarsLM 主模型（组装所有模块）
# =========================================================================
# 这是最顶层，把所有子模块组装成一个完整的 Decoder-Only 语言模型。
#
# 数据流（从输入到输出）：
#   input_ids (token 索引)
#       |
#   Embedding（把整数 token ID 映射为稠密向量）
#       |
#   Dropout（训练时随机丢弃，防过拟合）
#       |
#   [DecoderLayer x N]（N 层 Transformer，逐层处理）
#       |
#   RMSNorm（最终归一化）
#       |
#   lm_head（线性投影 hidden -> vocab_size，得到每个 token 的"得分"）
#       |
#   logits (batch, seq_len, vocab_size)
#
# 继承关系说明：
#   - PreTrainedModel：HuggingFace 基类，提供 save_pretrained / from_pretrained 等通用能力
#   - GenerationMixin：提供 model.generate() 方法，支持贪心解码、采样、beam search 等生成策略

class TarsLMModel(PreTrainedModel, GenerationMixin):
    """TarsLM Decoder-Only 语言模型。

    调用方式：
        from model.config import TarsLMConfig
        from model.model import TarsLMModel

        config = TarsLMConfig()          # 使用默认超参
        model = TarsLMModel(config)      # 实例化模型

        # 训练：输入 input_ids 和 labels，自动计算交叉熵损失
        loss = model(input_ids, labels=labels).loss

        # 推理：调用 generate() 逐 token 生成
        output_ids = model.generate(input_ids, max_new_tokens=100)
    """

    config_class = TarsLMConfig  # 文件内无显式引用, 由 PreTrainedModel 基类通过 type(self).config_class 访问, 用于 save/from_pretrained

    def __init__(self, config):
        """参数:
            config: TarsLMConfig 实例，包含所有架构超参
        """
        super().__init__(config)

        # ---- 1. 词嵌入层 ----
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                         padding_idx=config.pad_token_id)
        # ---- 2. Transformer 层堆叠 ----
        self.layers = nn.ModuleList([TarsLMDecoderLayer(config) for _ in range(config.num_layers)])
        # ---- 3. 最终归一化层 ----
        self.norm = TarsLMRMSNorm(config.hidden_size, config.norm_eps)
        # ---- 4. 语言模型头 ----
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.dropout = nn.Dropout(config.hidden_dropout)

        # ---- 5. 权重绑定（Weight Tying）----
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # ---- 6. HF 标准后处理 ----
        self.post_init()
        if config.tie_word_embeddings:
            # post_init 遍历到 lm_head 时可能覆盖共享的 embedding，重新保证 pad token 为零。
            with torch.no_grad():
                self.embed_tokens.weight[config.pad_token_id].zero_()

        # ---- 7. 默认生成参数 ----

    def _init_weights(self, module):
        """HuggingFace 标准权重初始化回调。"""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx] = 0.0

    def count_parameters(self) -> tuple[int, str]:
        """统计模型参数量并返回人类可读的缩写形式。

        Returns:
            (total_params, size_str)：例如 (1_300_000, "1.30M")。
        """
        total_params = sum(p.numel() for p in self.parameters())
        if total_params >= 1e9:
            size_str = f"{total_params / 1e9:.2f}B"
        elif total_params >= 1e6:
            size_str = f"{total_params / 1e6:.2f}M"
        elif total_params >= 1e3:
            size_str = f"{total_params / 1e3:.1f}K"
        else:
            size_str = f"{total_params:,}"
        return total_params, size_str

    def get_input_embeddings(self):
        """返回输入嵌入层。"""
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """替换输入嵌入层。"""
        self.embed_tokens = value

    def gradient_checkpointing_enable(self, **kwargs):
        """开启梯度检查点。"""
        for layer in self.layers:
            layer._gradient_checkpointing = True

    def gradient_checkpointing_disable(self, **kwargs):
        """关闭梯度检查点。"""
        for layer in self.layers:
            layer._gradient_checkpointing = False

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      attention_mask=None, position_ids=None,
                                      cache_position=None, **kwargs):
        """为增量推理准备输入。"""
        # DynamicCache is truthy even when empty; check actual content
        if past_key_values is not None:
            if hasattr(past_key_values, "get_seq_length"):
                has_cache = past_key_values.get_seq_length() > 0
            else:
                has_cache = len(past_key_values) > 0  # legacy tuple
        else:
            has_cache = False
        if has_cache:
            input_ids = input_ids[:, -1:]  # 只取当前步
            if position_ids is not None and position_ids.shape[-1] > input_ids.shape[-1]:
                position_ids = position_ids[:, -input_ids.shape[-1]:]
            if attention_mask is not None and attention_mask.dim() == 4:
                # 2D mask 保留完整长度，Attention 内部会结合缓存长度构造 key mask；
                # 4D mask 只裁掉当前 query 对应的历史行。
                attention_mask = attention_mask[:, :, -input_ids.shape[1]:, :]

        if position_ids is None:
            if cache_position is not None:
                position_ids = cache_position.unsqueeze(0).expand(input_ids.shape[0], -1)
            else:
                past_length = 0
                if past_key_values is not None:
                    if hasattr(past_key_values, "get_seq_length"):
                        past_length = past_key_values.get_seq_length()
                    elif len(past_key_values) > 0 and past_key_values[0][0] is not None:
                        past_length = past_key_values[0][0].shape[-2]
                position_ids = past_length + torch.arange(
                    input_ids.shape[1], device=input_ids.device, dtype=torch.long
                )
                position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)

        return {'input_ids': input_ids, 'past_key_values': past_key_values,
                'attention_mask': attention_mask, 'position_ids': position_ids,
                'use_cache': True}

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                past_key_values=None, use_cache=False, position_ids=None, **kwargs):
        """前向传播。"""
        # 兼容 HF >=4.45 的 DynamicCache 格式
        if isinstance(past_key_values, DynamicCache):
            past_key_values = [(layer.keys, layer.values) for layer in past_key_values.layers]

        # ---- 第 1 步：Token 嵌入 ----
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 如果开启了 KV 缓存，用列表收集每一层的 (k, v)
        new_kv_caches = [] if use_cache else None

        # ---- 第 2 步：逐层 Transformer ----
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if (past_key_values is not None
                                             and i < len(past_key_values)) else None
            hidden_states, kv_cache = layer(
                hidden_states, attention_mask=attention_mask,
                past_key_value=past_kv, use_cache=use_cache,
                position_ids=position_ids,
            )
            if use_cache:
                new_kv_caches.append(kv_cache)

        # ---- 第 3 步：最终 Norm + lm_head ----
        logits = self.lm_head(self.norm(hidden_states))

        # ---- 第 4 步：计算损失（如果提供了 labels）----
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1), ignore_index=-100,
                                   reduction='mean')
            if self.training:
                load_balance_loss = 0.0
                router_z_loss = 0.0
                for layer in self.layers:
                    if isinstance(layer.mlp, TarsLMMoE) and layer.mlp.aux_loss is not None:
                        layer_load_balance, layer_router_z = layer.mlp.aux_loss
                        load_balance_loss = load_balance_loss + layer_load_balance
                        router_z_loss = router_z_loss + layer_router_z
                loss = loss + (
                    self.config.moe_load_balance_weight * load_balance_loss
                    + self.config.moe_router_z_loss_weight * router_z_loss
                )

        return CausalLMOutputWithPast(
            loss=loss, logits=logits,
            past_key_values=tuple(new_kv_caches) if new_kv_caches is not None else None,
        )

    @property
    def _tied_weights_keys(self):
        """识别模型中的权重共享关系。"""
        if self.config.tie_word_embeddings:
            return {"lm_head.weight": "embed_tokens.weight"}
        return {}
