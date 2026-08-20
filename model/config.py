"""TarsLM 模型配置。

定义 vocab_size、hidden_size、num_layers 等全部架构超参，
继承 Hugging Face PretrainedConfig 以兼容 AutoModel 等生态接口。
"""

from transformers import PretrainedConfig


class TarsLMConfig(PretrainedConfig):
    # model_type 用于 AutoModel 自动识别并加载本配置类
    model_type = "tarslm"

    def __init__(
            self,
            vocab_size: int = 8000,               # 词表大小，需与分词器实际词表一致
            hidden_size: int = 256,               # Transformer 隐藏层维度
            num_layers: int = 6,                  # Decoder 层数（depth）
            num_attention_heads: int = 8,         # 多头注意力头数（MHA）
            num_key_value_heads: int = 8,         # KV 头数（GQA 支持；等于 num_attention_heads 时为 MHA）
            intermediate_size: int = 1024,        # FFN / SwiGLU 中间层维度
            max_seq_len: int = 1024,              # 模型支持的最大序列长度（位置编码预计算上限）
            norm_eps: float = 1e-6,               # RMSNorm 的 epsilon，防除零
            attention_dropout: float = 0.0,       # 注意力 dropout 率（对齐 Llama 默认值 0）
            hidden_dropout: float = 0.1,          # 残差路径 dropout 率
            use_bias: bool = False,               # 线性投影是否使用 bias（LLM 通常为 False）
            use_moe: bool = True,                 # 是否用 MoE 替代标准 FFN
            num_experts: int = 2,                 # MoE 专家总数
            moe_top_k: int = 1,                   # MoE 路由中每个 token 激活的专家数
            moe_load_balance_weight: float = 0.01,  # MoE 负载均衡辅助损失权重
            moe_router_z_loss_weight: float = 0.0,  # MoE Router z-loss 权重
            rope_theta: float = 10000.0,          # RoPE 旋转基频（越大支持越长外推）
            initializer_range: float = 0.02,      # 权重初始化标准差（对齐 Llama）
            tie_word_embeddings: bool = True,     # 是否共享输入嵌入层和 lm_head 权重
            pad_token_id: int = 0,                # 填充 token ID，训练时以 tokenizer 实际值为准
            bos_token_id: int = 1,                # 句首 token ID，训练时以 tokenizer 实际值为准
            eos_token_id: int = 2,                # 句尾 token ID，训练时以 tokenizer 实际值为准
            **kwargs,
    ):
        # 早期校验，避免后续 reshape/索引阶段才暴露难懂错误。
        if vocab_size <= 0:
            raise ValueError("vocab_size 必须大于 0")
        if hidden_size <= 0 or num_layers <= 0 or intermediate_size <= 0:
            raise ValueError("hidden_size、num_layers、intermediate_size 必须大于 0")
        if num_attention_heads <= 0 or num_key_value_heads <= 0:
            raise ValueError("注意力头数必须大于 0")
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size 必须能被 num_attention_heads 整除")
        if (hidden_size // num_attention_heads) % 2 != 0:
            raise ValueError("每个注意力头的维度必须是偶数")
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError("num_attention_heads 必须能被 num_key_value_heads 整除")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len 必须大于 0")
        if norm_eps <= 0 or rope_theta <= 0:
            raise ValueError("norm_eps 和 rope_theta 必须大于 0")
        if not 0 <= pad_token_id < vocab_size:
            raise ValueError("pad_token_id 必须在 [0, vocab_size) 范围内")
        if not 0 <= bos_token_id < vocab_size:
            raise ValueError("bos_token_id 必须在 [0, vocab_size) 范围内")
        if not 0 <= eos_token_id < vocab_size:
            raise ValueError("eos_token_id 必须在 [0, vocab_size) 范围内")
        if use_moe:
            if num_experts <= 0 or moe_top_k <= 0:
                raise ValueError("启用 MoE 时 num_experts 和 moe_top_k 必须大于 0")
            if moe_top_k > num_experts:
                raise ValueError("moe_top_k 不能大于 num_experts")
            if moe_load_balance_weight < 0 or moe_router_z_loss_weight < 0:
                raise ValueError("MoE 辅助损失权重不能为负数")

        # 将特殊 token ID 传给父类，确保 generation / tokenizer 对齐
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        # 以下将全部架构超参挂载到 self 上，供建模代码通过 config.xxx 访问
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_hidden_layers = num_layers  # HF 标准别名（generate / cache 需要）
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len
        self.max_position_embeddings = max_seq_len
        self.norm_eps = norm_eps
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.use_bias = use_bias
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.moe_top_k = moe_top_k
        self.moe_load_balance_weight = moe_load_balance_weight
        self.moe_router_z_loss_weight = moe_router_z_loss_weight
        self.rope_theta = rope_theta
        self.initializer_range = initializer_range
