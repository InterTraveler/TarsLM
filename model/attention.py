"""多头自注意力模块，支持 MHA / GQA 与 PyTorch SDPA。

计算流程：Q/K/V 线性投影 → RoPE 位置编码 → KV Cache 拼接 →
GQA 展开 → 掩码处理 → PyTorch SDPA → O 投影合并。
"""

import torch
import torch.nn.functional as F
from torch import nn

from model.rope import TarsLMRotaryEmbedding


class TarsLMAttention(nn.Module):
    """多头自注意力模块（MHA / GQA）。

    GQA 原理: 多个 Q 头共享一组 KV 头。当 num_kv_heads < num_heads 时，
    每 num_key_value_groups 个 Q 头共享 1 个 KV 头对，KV Cache 显存节省 n 倍
    （Llama 3 / Mistral / Gemma 2 标配）。

    num_kv_heads == num_heads 时退化为标准 MHA, _repeat_kv 为 no-op。
    """

    def __init__(self, config):
        super().__init__()

        # ---- 头部配置 ----
        self.num_heads = config.num_attention_heads  # Q 头数
        self.num_kv_heads = config.num_key_value_heads  # KV 头数（GQA 核心参数）
        self.num_key_value_groups = self.num_heads // self.num_kv_heads  # 每组 Q 头数
        self.head_dim = config.hidden_size // config.num_attention_heads  # 每头维度
        self.attention_dropout = config.attention_dropout  # 注意力 dropout 率

        # ---- 线性投影层（全部无 bias: LLM 中 bias 不提升效果, 浪费参数/显存） ----
        # Q 投影: "我要查什么？" —— 始终 num_heads 个头
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.use_bias)
        # K 投影: "我有什么标签？" —— GQA 用 num_kv_heads 个头（少于 Q）
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.use_bias)
        # V 投影: "我实际包含什么信息？" —— 同 K
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.use_bias)
        # O 投影: "把多头的结果合并回 hidden_size"
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.use_bias)

        # ---- 旋转位置编码器 ----
        self.rotary_emb = TarsLMRotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)

    def _repeat_kv(self, x):
        """GQA 核心操作: 将 KV 头沿组维度复制，从 num_kv_heads 扩展到 num_heads。

        操作流程（例: num_kv_heads=2, num_key_value_groups=4）:
          (bsz, 2, seq, dim) -> (bsz, 2, 1, seq, dim) -> expand groups
          -> (bsz, 2, 4, seq, dim) -> reshape -> (bsz, 8, seq, dim)

        num_key_value_groups == 1 时（MHA 模式），短路直接返回。
        """
        if self.num_key_value_groups == 1:
            return x  # MHA 模式，无需展开
        bsz, num_kv_heads, seq_len, head_dim = x.shape
        # 插入组维度 → expand（不复制内存，仅改变 stride）→ reshape 合并维度
        x = x[:, :, None, :, :].expand(bsz, num_kv_heads, self.num_key_value_groups, seq_len, head_dim)
        return x.reshape(bsz, num_kv_heads * self.num_key_value_groups, seq_len, head_dim)

    def forward(self, hidden_states, attention_mask=None, past_key_value=None, use_cache=False,
                position_ids=None):
        """前向传播 —— 完整的 7 步注意力计算。

        参数:
            hidden_states:  (batch, seq_len, hidden_size)
            attention_mask: 注意力掩码（None = 纯因果掩码; 2D/4D = padding 掩码）
            past_key_value: KV 缓存（增量推理时复用历史 K、V）
            use_cache:      是否返回 KV 缓存（增量推理时需要）
        返回:
            (attn_output, kv_cache)
            attn_output: (batch, seq_len, hidden_size)
            kv_cache:    (k, v) 或 None
        """
        bsz, seq_len, _ = hidden_states.shape

        # ---- 步骤 1: 线性投影 + 重塑为多头格式 ----
        # .view(bsz, seq, heads, head_dim) → 把 hidden_size 拆成 heads x head_dim
        # .transpose(1, 2) → 把 heads 维度移到 seq 前: (bsz, heads, seq, dim)
        #   这是 SDPA 函数要求的输入格式

        # Q: (batch, num_heads, seq, head_dim)
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # K/V: (batch, num_kv_heads, seq, head_dim) —— GQA 比 MHA 少
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # ---- 步骤 2: 施加 RoPE 位置编码（只对 Q 和 K，不对 V） ----
        q, k = self.rotary_emb(q, k, position_ids=position_ids)

        # ---- 步骤 3: 拼接历史 KV 缓存（增量推理核心） ----
        # 推理时每次只输入 1 个新 token，需把之前所有 token 的 K、V 拼回来，
        # 这样 Attention 才能"看到"完整的上文。这就是 KV Cache 的核心作用。
        if past_key_value is not None and past_key_value[0] is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-2)  # 沿序列维度拼接: 旧 K + 新 K
            v = torch.cat([past_v, v], dim=-2)

        # ---- 步骤 3.5: 保存 KV 缓存（展开之前，省显存） ----
        # GQA 模式下缓存的 K/V 是未展开的 num_kv_heads 份，而非全量 num_heads 份
        kv_cache = (k, v) if use_cache else None

        # ---- 步骤 4: GQA 展开 KV 头以匹配 Q 头 ----
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # ---- 步骤 5: 统一 attention_mask 格式 ----
        # HF 框架可能传入 2D (batch, seq) 或 4D (batch, 1, seq, seq) 的 mask
        # SDPA 要求 4D additive mask: 0(关注) / -inf(屏蔽)
        needs_causal_mask = attention_mask is None
        if attention_mask is None and q.shape[-2] < k.shape[-2]:
            # 增量推理时 query 对应尾部 token，需要按 past_len 偏移构造因果 mask。
            query_len, key_len = q.shape[-2], k.shape[-2]
            past_len = key_len - query_len
            query_idx = torch.arange(query_len, device=q.device)[:, None]
            key_idx = torch.arange(key_len, device=q.device)[None, :]
            causal_mask = key_idx > (query_idx + past_len)
            causal_mask = causal_mask[None, None, :, :].to(q.dtype)
            attention_mask = causal_mask * torch.finfo(q.dtype).min
            needs_causal_mask = False

        if attention_mask is not None:
            is_2d_mask = attention_mask.dim() == 2
            if is_2d_mask:
                # 增量推理时 attention_mask 可能只覆盖当前 query，补齐历史 key 的 mask。
                if past_key_value is not None and past_key_value[0] is not None:
                    past_len = past_key_value[0].shape[-2]
                    if attention_mask.shape[-1] < past_len + seq_len:
                        prefix = torch.ones(
                            *attention_mask.shape[:-1],
                            past_len + seq_len - attention_mask.shape[-1],
                            device=attention_mask.device,
                            dtype=attention_mask.dtype,
                        )
                        attention_mask = torch.cat([prefix, attention_mask], dim=-1)
                # 2D → 4D: 在中间插入两个大小为 1 的维度以适配 SDPA 接口
                attention_mask = attention_mask[:, None, None, :].to(dtype=q.dtype)
                # 将 0/1 mask 转为 SDPA additive mask: 0(屏蔽) → -inf, 1(关注) → 0.0
                attention_mask = (1.0 - attention_mask) * torch.finfo(q.dtype).min
            elif attention_mask.dtype not in (torch.bool, torch.float16, torch.float32, torch.bfloat16):
                attention_mask = attention_mask.to(q.dtype)

            if is_2d_mask:
                query_len, key_len = q.shape[-2], k.shape[-2]
                past_len = key_len - query_len
                query_idx = torch.arange(query_len, device=q.device)[:, None]
                key_idx = torch.arange(key_len, device=q.device)[None, :]
                causal_mask = key_idx > (query_idx + past_len)
                causal_mask = causal_mask[None, None, :, :].to(q.dtype)
                causal_mask = causal_mask * torch.finfo(q.dtype).min
                attention_mask = attention_mask + causal_mask

        # ---- 步骤 6: 执行注意力计算 ----
        # PyTorch SDPA 会在受支持的 GPU 上自动选择 FlashAttention 后端，
        # 既能处理 padding mask，也符合 HF 生态的推荐实现方式。
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=needs_causal_mask,
        )
        # SDPA 返回 (batch, heads, seq, head_dim)，需要转置回 (batch, seq, heads, dim)
        attn_output = attn_output.transpose(1, 2)

        # ---- 步骤 7: 合并多头 → O 投影 ----
        # .contiguous(): 保证内存连续, .view() 才能安全执行
        attn_output = attn_output.contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output), kv_cache
