"""TarsLMModel 端到端单元测试。"""

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from model.config import TarsLMConfig
from model.model import TarsLMModel
from tests.utils import assert_tensors_allclose


@pytest.fixture
def model(model_config):
    """创建标准 TarsLMModel"""
    return TarsLMModel(model_config)


class TestShape:
    """验证所有输出张量形状正确"""

    def test_logits(self, model):
        """logits: (batch, seq, vocab_size)"""
        model.eval()
        out = model(torch.randint(0, 128, (2, 16)))
        assert out.logits.shape == (2, 16, 128)

    def test_loss_is_scalar(self, model):
        """有 labels 时 loss 是标量"""
        model.train()
        ids = torch.randint(1, 128, (2, 16))
        assert model(ids, labels=ids).loss.dim() == 0

    def test_kv_cache(self, model):
        """use_cache=True 返回 num_layers 个 KV 对"""
        model.eval()
        out = model(torch.randint(0, 128, (1, 8)), use_cache=True)
        assert out.past_key_values is not None and len(out.past_key_values) == 2


class TestGeneration:
    """验证 model.generate() 基本功能"""

    def test_shape(self, model):
        """生成后序列长度 = prompt_len + max_new_tokens"""
        model.eval()
        out = model.generate(torch.randint(0, 128, (1, 4)), max_new_tokens=5, do_sample=False)
        assert out.shape == (1, 9)

    def test_deterministic(self, model):
        """贪心解码对相同输入产生相同输出"""
        model.eval()
        ids = torch.randint(0, 128, (1, 4))
        o1 = model.generate(ids, max_new_tokens=3, do_sample=False)
        o2 = model.generate(ids, max_new_tokens=3, do_sample=False)
        assert torch.equal(o1, o2)


class TestLoss:
    """验证自回归损失计算正确性"""

    def test_not_nan(self, model):
        """正常训练不产生 NaN/Inf 损失"""
        model.train()
        ids = torch.randint(1, 128, (2, 8))
        loss = model(ids, labels=ids).loss
        assert not torch.isnan(loss) and not torch.isinf(loss)

    def test_ignore_index(self, model):
        """ignore_index=-100 的 token 被忽略后损失与不忽略时不同"""
        model.train()
        ids = torch.randint(1, 128, (2, 8))
        labels_partial = ids.clone()
        labels_partial[:, :4] = -100
        l1 = model(ids, labels=ids).loss.item()
        l2 = model(ids, labels=labels_partial).loss.item()
        assert abs(l1 - l2) > 1e-8

    def test_expected_initial_loss(self):
        """随机初始化时交叉熵约 ln(vocab_size)。偏离超 20% 说明初始化或 loss 有 bug。"""
        cfg = TarsLMConfig(vocab_size=128, hidden_size=64, num_layers=1,
                           num_attention_heads=4, num_key_value_heads=4,
                           intermediate_size=256, max_seq_len=32,
                           attention_dropout=0.0, hidden_dropout=0.0,
                           use_moe=False, tie_word_embeddings=False,
                           pad_token_id=0, bos_token_id=1, eos_token_id=2)
        m = TarsLMModel(cfg)
        m.train()
        ids = torch.randint(1, cfg.vocab_size, (2, 16))
        loss = m(ids, labels=ids).loss.item()
        expected = math.log(cfg.vocab_size)  # ln(128) ~ 4.85
        assert 0.8 * expected < loss < 1.2 * expected, (
            f"Initial loss {loss:.3f} far from expected {expected:.3f}"
        )


class TestOverfit:
    """业界金标准: 模型必须能过拟合极小数据集"""

    def test_overfit_tiny_batch(self):
        """单样本 8 token, 训练 100 步后 loss 降至初始 30% 以下。
        如果不能, 说明架构存在根本缺陷 (Karpathy 方法论)。"""
        cfg = TarsLMConfig(vocab_size=128, hidden_size=32, num_layers=1,
                           num_attention_heads=2, num_key_value_heads=2,
                           intermediate_size=128, max_seq_len=8,
                           attention_dropout=0.0, hidden_dropout=0.0,
                           tie_word_embeddings=False,
                           pad_token_id=0, bos_token_id=1, eos_token_id=2)
        m = TarsLMModel(cfg)
        m.train()
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
        labels = ids.clone()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
        for step in range(100):
            opt.zero_grad()
            loss = m(ids, labels=labels).loss
            loss.backward()
            opt.step()
            if step == 0: initial_loss = loss.item()
        assert loss.item() < initial_loss * 0.3, (
            f"Overfit failed: initial={initial_loss:.3f}, final={loss.item():.3f}"
        )


class TestLossMean:
    """默认损失应为每 token 平均，而不是未归一化的求和。"""

    def test_loss_matches_manual_mean(self):
        """模型内置 loss 必须等于手工 shift 后的逐 token 平均交叉熵。

        关闭 hidden_dropout 以保证两次前向完全确定（可逐位对比）。
        """
        cfg = TarsLMConfig(vocab_size=128, hidden_size=64, num_layers=2,
                           num_attention_heads=8, num_key_value_heads=8,
                           intermediate_size=256, max_seq_len=32,
                           attention_dropout=0.0, hidden_dropout=0.0,
                           use_moe=False, tie_word_embeddings=False)
        model = TarsLMModel(cfg)
        ids = torch.randint(1, 128, (4, 32))
        loss_model = model(ids, labels=ids).loss.item()
        with torch.no_grad():
            logits = model(ids).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = ids[:, 1:].contiguous()
        loss_mean = F.cross_entropy(shift_logits.view(-1, 128),
                                    shift_labels.view(-1), ignore_index=-100).item()
        # 两条路径应逐位一致，容差收紧到数值精度级别
        assert abs(loss_model - loss_mean) < 1e-4


class TestWeightTying:
    """验证 lm_head 与 embed_tokens 权重共享"""

    def test_same_memory(self):
        """tie_word_embeddings=True 时两者指向同一块内存"""
        cfg = TarsLMConfig(vocab_size=128, hidden_size=64, num_layers=1,
                           num_attention_heads=4, num_key_value_heads=4,
                           intermediate_size=256, max_seq_len=16,
                           tie_word_embeddings=True)
        m = TarsLMModel(cfg)
        assert m.lm_head.weight.data_ptr() == m.embed_tokens.weight.data_ptr()

    def test_gradient_accumulates(self):
        """共享权重时 embedding 梯度正确累积 (不为 None)"""
        cfg = TarsLMConfig(vocab_size=128, hidden_size=64, num_layers=1,
                           num_attention_heads=4, num_key_value_heads=4,
                           intermediate_size=256, max_seq_len=16,
                           tie_word_embeddings=True)
        m = TarsLMModel(cfg)
        m.train()
        ids = torch.randint(1, 128, (2, 8))
        m(ids, labels=ids).loss.backward()
        assert m.embed_tokens.weight.grad is not None


class TestInit:
    """验证模型初始化正确性"""

    def test_padding_embedding_is_zero(self):
        """<pad> token 嵌入必须全零 (修复 _init_weights 覆盖 bug 后的回归)"""
        cfg = TarsLMConfig(vocab_size=128, hidden_size=64, num_layers=1,
                           num_attention_heads=4, num_key_value_heads=4,
                           intermediate_size=256, max_seq_len=16,
                           tie_word_embeddings=False, pad_token_id=0)
        m = TarsLMModel(cfg)
        pad = m.embed_tokens.weight[cfg.pad_token_id]
        assert_tensors_allclose(pad, torch.zeros_like(pad))

    def test_padding_embedding_is_zero_when_tied(self):
        """tie_word_embeddings=True 时，共享 lm_head 也不能覆盖 pad 零初始化。"""
        cfg = TarsLMConfig(vocab_size=128, hidden_size=64, num_layers=1,
                           num_attention_heads=4, num_key_value_heads=4,
                           intermediate_size=256, max_seq_len=16,
                           tie_word_embeddings=True, pad_token_id=0)
        m = TarsLMModel(cfg)
        pad = m.embed_tokens.weight[cfg.pad_token_id]
        assert_tensors_allclose(pad, torch.zeros_like(pad))

    def test_no_nan_in_first_forward(self, model):
        """模型创建后第一次前向不应有 NaN"""
        model.eval()
        out = model(torch.randint(0, 128, (2, 8)))
        assert not torch.isnan(out.logits).any()


class TestGradient:
    """端到端梯度完整性"""

    def test_full_backward(self, model):
        """训练时所有参数都有梯度"""
        model.train()
        ids = torch.randint(1, 128, (2, 8))
        model(ids, labels=ids).loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"{name} missing grad"

    def test_checkpointing(self, model):
        """梯度检查点前后损失和梯度范数一致"""
        model.train()
        ids = torch.randint(1, 128, (2, 8))
        loss1 = model(ids, labels=ids).loss
        loss1.backward()
        gn1 = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        model.zero_grad()
        model.gradient_checkpointing_enable()
        loss2 = model(ids, labels=ids).loss
        loss2.backward()
        gn2 = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        assert abs(loss1.item() - loss2.item()) < 1e-4
        assert abs(gn1 - gn2) / max(gn1, 1e-8) < 1e-2


class TestKVCacheNumerical:
    """端到端 KV 缓存数值一致性。

    静默 bug：任意一层的 KV 缓存维度不匹配或拼接顺序错误，
    都会导致增量解码的 logits 出现静默偏差。本测试验证全模型
    级别的增量推理形状正确、无 NaN。
    """

    def test_full_vs_incremental(self):
        """逐 token 增量解码应与全量前向数值一致。"""
        cfg = TarsLMConfig(vocab_size=128, hidden_size=64, num_layers=2,
                           num_attention_heads=4, num_key_value_heads=4,
                           intermediate_size=256, max_seq_len=16,
                           attention_dropout=0.0, hidden_dropout=0.0,
                           tie_word_embeddings=False,
                           pad_token_id=0, bos_token_id=1, eos_token_id=2)
        model = TarsLMModel(cfg)
        model.eval()
        input_ids = torch.randint(1, 128, (1, 8))
        # 全量前向（8 token 一次性）
        position_ids = torch.arange(8, dtype=torch.long).unsqueeze(0)
        out_full = model(input_ids, position_ids=position_ids, use_cache=False)
        logits_full = out_full.logits
        # 增量模式（逐 token 前向，拼接 KV 缓存）
        past_kv = None
        logits_inc = []
        for i in range(8):
            step_input = input_ids[:, i:i + 1]
            out_step = model(
                step_input,
                past_key_values=past_kv,
                position_ids=position_ids[:, i:i + 1],
                use_cache=True,
            )
            past_kv = out_step.past_key_values
            logits_inc.append(out_step.logits)
        logits_inc = torch.cat(logits_inc, dim=1)
        assert logits_inc.shape == (1, 8, 128)
        assert not torch.isnan(logits_inc).any()
        assert len(past_kv) == 2
        assert past_kv[0][0].shape[-2] == 8
        assert_tensors_allclose(logits_inc, logits_full, rtol=1e-4, atol=1e-5)

    def test_full_vs_prefix_then_batch(self):
        """前缀缓存 + 批量续推：输出形状正确，无 NaN。"""
        cfg = TarsLMConfig(vocab_size=128, hidden_size=64, num_layers=2,
                           num_attention_heads=4, num_key_value_heads=4,
                           intermediate_size=256, max_seq_len=16,
                           attention_dropout=0.0, hidden_dropout=0.0,
                           tie_word_embeddings=False,
                           pad_token_id=0, bos_token_id=1, eos_token_id=2)
        model = TarsLMModel(cfg)
        model.eval()
        input_ids = torch.randint(1, 128, (1, 8))
        # 前缀（前 4 个 token，带 KV 缓存产出）
        out_pref = model(input_ids[:, :4], use_cache=True)
        # 批量续推（后 4 个 token，复用前缀的 KV 缓存）
        out_batch = model(input_ids[:, 4:], past_key_values=out_pref.past_key_values, use_cache=False)
        logits_combined = torch.cat([out_pref.logits, out_batch.logits], dim=1)
        assert logits_combined.shape == (1, 8, 128)
        assert not torch.isnan(logits_combined).any()


class TestLossShift:
    """自回归损失 shift 方向正确性：logits 左移一位预测 labels 右移一位。

    自回归语言模型的核心公式：位置 i 的 logits 应预测位置 i+1 的 token。
    shift 方向反了（右移 logits 对左移 labels）会产生"看起来正常"的损失
    值，但模型实际上学到了错误的预测目标——这是最危险的静默 bug 之一。
    """

    def test_shift_direction_manual(self):
        """手动计算 shift 并验证与模型 loss 完全一致。

        用已知的 input_ids 和 labels 构造确定性测试：
        手动取 logits[:, :-1] 和 labels[:, 1:] 计算交叉熵，
        与 model.forward 内置的 loss 逐位对比。
        静默 bug：shift 方向反了会产生完全不同的 loss 值。
        """
        cfg = TarsLMConfig(vocab_size=16, hidden_size=64, num_layers=1,
                           num_attention_heads=4, num_key_value_heads=4,
                           intermediate_size=256, max_seq_len=8,
                           attention_dropout=0.0, hidden_dropout=0.0,
                           use_moe=False, tie_word_embeddings=False,
                           pad_token_id=0, bos_token_id=1, eos_token_id=2)
        model = TarsLMModel(cfg)
        model.train()
        # 构造确定性测试：用固定值而非随机值
        input_ids = torch.tensor([[1, 3, 5, 7, 2, 6, 4, 9]], dtype=torch.long)
        labels = torch.tensor([[2, 6, 4, 9, 1, 3, 5, 7]], dtype=torch.long)
        loss_model = model(input_ids, labels=labels).loss
        # 手动执行 shift：logits 截取 [..., :-1, :]，labels 截取 [..., 1:]
        with torch.no_grad():
            logits = model(input_ids).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_manual = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), ignore_index=-100,
            reduction='mean')
        assert abs(loss_model.item() - loss_manual.item()) < 1e-6, (
            f"Loss shift mismatch: model={loss_model.item():.6f}, manual={loss_manual.item():.6f}"
        )

    def test_labels_all_ignored(self):
        """labels 全为 -100 时前向不崩溃，logits 无 NaN。

        当所有 token 都被标注为 ignore_index=-100 时，交叉熵返回 NaN
        （PyTorch 标准行为）。关键不变性：前向必须不崩溃，logits 必须有效。
        """
        cfg = TarsLMConfig(vocab_size=16, hidden_size=64, num_layers=1,
                           num_attention_heads=4, num_key_value_heads=4,
                           intermediate_size=256, max_seq_len=8,
                           attention_dropout=0.0, hidden_dropout=0.0,
                           tie_word_embeddings=False,
                           pad_token_id=0, bos_token_id=1, eos_token_id=2)
        model = TarsLMModel(cfg)
        model.train()
        input_ids = torch.randint(1, 8, (2, 8))
        labels = torch.full_like(input_ids, -100)
        out = model(input_ids, labels=labels)
        # 全部被忽略时交叉熵返回 NaN（PyTorch 标准行为）。
        # 关键不变性：前向必须不崩溃，logits 必须有效。
        assert not torch.isnan(out.logits).any()


class TestDynamicCache:
    """HuggingFace DynamicCache 格式兼容性。

    HF >= 4.45 的 generate() 会将 past_key_values 包装为 DynamicCache 对象。
    模型必须能接受并正确处理此格式，否则生成时会崩溃。
    """

    def test_dynamic_cache_accepts(self, model):
        """模型前向必须能接受 HF DynamicCache 作为 past_key_values。

        显式传入 position_ids，确保增量步使用正确的绝对位置。
        """
        from transformers.cache_utils import DynamicCache
        model.eval()
        input_ids = torch.randint(0, 128, (1, 4))
        out1 = model(input_ids, use_cache=True,
                     position_ids=torch.arange(4).unsqueeze(0))
        # 将 tuple 格式转为 DynamicCache
        dc = DynamicCache()
        for layer_idx, (k, v) in enumerate(out1.past_key_values):
            dc.update(k, v, layer_idx)
        # 用 DynamicCache 做第二步（固定 token + 显式位置 4）
        out2 = model(torch.tensor([[50]]),
                     past_key_values=dc, use_cache=True,
                     position_ids=torch.tensor([[4]]))
        assert out2.logits.shape == (1, 1, 128)

    def test_dynamic_cache_yields_same_logits(self, model):
        """tuple 缓存与 DynamicCache 缓存产生的 logits 完全一致。

        显式传入 position_ids，避免两条路径以相同方式使用错误的
        绝对位置（位置 0）而掩盖增量位置错位问题。
        """
        from transformers.cache_utils import DynamicCache
        model.eval()
        input_ids = torch.randint(0, 128, (1, 4))
        # 第一步：完整 prompt，显式位置 0-3
        out1 = model(input_ids, use_cache=True,
                     position_ids=torch.arange(4).unsqueeze(0))
        # 第二步：增量 token，使用 tuple 缓存
        next_token = torch.tensor([[50]])
        out2_tuple = model(next_token, past_key_values=out1.past_key_values,
                           use_cache=True, position_ids=torch.tensor([[4]]))
        # 第二步：增量 token，使用 DynamicCache 缓存
        dc = DynamicCache()
        for layer_idx, (k, v) in enumerate(out1.past_key_values):
            dc.update(k, v, layer_idx)
        out2_dc = model(next_token, past_key_values=dc, use_cache=True,
                        position_ids=torch.tensor([[4]]))
        assert_tensors_allclose(out2_tuple.logits, out2_dc.logits, rtol=1e-4,
                                msg="tuple 缓存与 DynamicCache 缓存产生的 logits 不一致")


class TestSaveLoad:
    """save_pretrained / from_pretrained 保存加载往返测试。"""

    def test_save_and_load_weights_match(self):
        """保存后重新加载，所有权重参数数值完全一致。

        用 save_pretrained 保存到临时目录，再用 from_pretrained 加载，
        逐参数比较原始模型与加载模型的权重值。
        """
        import tempfile
        cfg = TarsLMConfig(vocab_size=64, hidden_size=32, num_layers=1,
                           num_attention_heads=4, num_key_value_heads=4,
                           intermediate_size=128, max_seq_len=16,
                           attention_dropout=0.0, hidden_dropout=0.0,
                           tie_word_embeddings=False,
                           pad_token_id=0, bos_token_id=1, eos_token_id=2)
        model1 = TarsLMModel(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            model1.save_pretrained(tmpdir)
            model2 = TarsLMModel.from_pretrained(tmpdir)
        model2.eval()
        # 验证重新加载后所有参数数值完全一致
        for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            assert torch.allclose(p1, p2, rtol=1e-5, atol=1e-5), (
                f"Parameter {n1} differs after save/load"
            )


class TestHFInterface:
    """HuggingFace 兼容接口测试。"""

    def test_get_set_input_embeddings(self, model):
        """get_input_embeddings 和 set_input_embeddings 往返一致性。

        先获取原始 embedding，替换为新 embedding 后验证 get 返回的是新对象，
        最后恢复原始 embedding。
        """
        orig = model.get_input_embeddings()
        new_emb = nn.Embedding(128, 64)
        model.set_input_embeddings(new_emb)
        assert model.get_input_embeddings() is new_emb
        model.set_input_embeddings(orig)

    def test_prepare_inputs_truncates(self, model):
        """prepare_inputs_for_generation 应截断输入并生成正确的绝对位置。"""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        past_kv = model(input_ids, use_cache=True).past_key_values
        prepared = model.prepare_inputs_for_generation(
            torch.tensor([[1, 2, 3, 4, 5, 6]]),  # 完整 prompt + 1 个新 token
            past_key_values=past_kv,
            attention_mask=torch.ones(1, 6))
        # 应只保留最后一个 token（id=6）
        assert prepared['input_ids'].shape == (1, 1)
        assert prepared['input_ids'][0, 0].item() == 6
        assert prepared['position_ids'].tolist() == [[5]]
        # 2D attention_mask 应保留完整长度，供 Attention 构造历史 key mask。
        assert prepared['attention_mask'].shape == (1, 6)


class TestGQAAndMoE:
    """GQA + MoE 组合模型的完整端到端测试。"""

    def test_gqa_moe_forward(self):
        """GQA(8/2) + MoE(2专家, top_k=1) 组合模型前向：形状正确、loss 非 NaN。"""
        cfg = TarsLMConfig(vocab_size=128, hidden_size=64, num_layers=2,
                           num_attention_heads=8, num_key_value_heads=2,  # GQA
                           intermediate_size=256, max_seq_len=32,
                           use_moe=True, num_experts=2, moe_top_k=1,  # MoE
                           attention_dropout=0.0, hidden_dropout=0.0,
                           tie_word_embeddings=False,
                           pad_token_id=0, bos_token_id=1, eos_token_id=2)
        model = TarsLMModel(cfg)
        model.train()
        input_ids = torch.randint(1, 128, (2, 16))
        labels = input_ids.clone()
        out = model(input_ids, labels=labels)
        assert out.logits.shape == (2, 16, 128)
        assert out.loss is not None and not torch.isnan(out.loss)

    def test_gqa_moe_gradient(self):
        """GQA + MoE 组合模型：所有参数都能收到梯度。

        静默 bug：GQA 的 _repeat_kv 中 expand/reshape 如果断开梯度图，
        或 MoE router 的 topk 操作不可导，会导致部分参数梯度为 None。
        """
        cfg = TarsLMConfig(vocab_size=128, hidden_size=64, num_layers=1,
                           num_attention_heads=8, num_key_value_heads=2,
                           intermediate_size=256, max_seq_len=16,
                           use_moe=True, num_experts=2, moe_top_k=1,
                           attention_dropout=0.0, hidden_dropout=0.0,
                           tie_word_embeddings=False,
                           pad_token_id=0, bos_token_id=1, eos_token_id=2)
        model = TarsLMModel(cfg)
        model.train()
        input_ids = torch.randint(1, 128, (2, 8))
        labels = input_ids.clone()
        model(input_ids, labels=labels).loss.backward()
        no_grad = [n for n, p in model.named_parameters() if p.grad is None]
        assert len(no_grad) == 0, f"Parameters without gradient: {no_grad}"

    def test_gqa_moe_kv_cache(self):
        """GQA + MoE 模型的 KV 缓存存储正确数量的头（num_kv_heads=2 而非 8）。

        验证全模型中每层 KV 缓存的 K 头数为 2，确保 GQA 显存节省在实际
        端到端流程中生效（而非仅在单层 attention 测试中）。
        """
        cfg = TarsLMConfig(vocab_size=128, hidden_size=64, num_layers=2,
                           num_attention_heads=8, num_key_value_heads=2,
                           intermediate_size=256, max_seq_len=32,
                           use_moe=True, num_experts=2, moe_top_k=1,
                           attention_dropout=0.0, hidden_dropout=0.0,
                           tie_word_embeddings=False,
                           pad_token_id=0, bos_token_id=1, eos_token_id=2)
        model = TarsLMModel(cfg)
        model.eval()
        out = model(torch.randint(0, 128, (1, 8)), use_cache=True)
        # 每层的 K 缓存头数应为 num_kv_heads=2
        for layer_kv in out.past_key_values:
            assert layer_kv[0].shape[1] == 2, (
                f"GQA KV cache should store 2 heads, got {layer_kv[0].shape[1]}"
            )
