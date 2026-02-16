"""
Unit Tests for Transformer Model Components
============================================

Tests cover shape correctness, mathematical properties, and
weight tying for all custom-built model components.

Run with::

    python -m pytest tests/ -v
"""

import torch
import torch.nn as nn
import pytest

from transformer.model import (
    Linear,
    Embedding,
    RMSNorm,
    SwiGLUFeedForward,
    RotaryPositionalEmbedding,
    MultiHeadAttention,
    EncoderBlock,
    DecoderBlock,
    Transformer,
    build_transformer,
    softmax,
    scaled_dot_product_attention,
    param_init,
)
from transformer.dataset import causal_mask


# ── Test Fixtures ────────────────────────────────────────────────────

BATCH_SIZE = 2
SEQ_LEN = 16
D_MODEL = 64
NUM_HEADS = 4
VOCAB_SIZE = 100


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def sample_input(device):
    """Random float tensor (B, T, D)."""
    return torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, device=device)


@pytest.fixture
def sample_ids(device):
    """Random integer token IDs (B, T)."""
    return torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)


# ── Primitive Layers ─────────────────────────────────────────────────


class TestLinear:
    def test_output_shape(self, sample_input):
        layer = Linear(D_MODEL, 128)
        layer.apply(param_init)
        output = layer(sample_input)
        assert output.shape == (BATCH_SIZE, SEQ_LEN, 128)

    def test_no_bias(self):
        layer = Linear(D_MODEL, 128)
        params = list(layer.parameters())
        assert len(params) == 1, "Linear should have exactly 1 parameter (W, no bias)"


class TestEmbedding:
    def test_output_shape(self, sample_ids):
        embed = Embedding(VOCAB_SIZE, D_MODEL)
        embed.apply(param_init)
        output = embed(sample_ids)
        assert output.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)

    def test_lookup_correctness(self):
        embed = Embedding(VOCAB_SIZE, D_MODEL)
        embed.apply(param_init)
        ids = torch.tensor([[0, 1, 2]])
        output = embed(ids)
        # Each output should match the corresponding row of the embedding matrix
        assert torch.allclose(output[0, 0], embed.embedding_matrix[0])
        assert torch.allclose(output[0, 1], embed.embedding_matrix[1])
        assert torch.allclose(output[0, 2], embed.embedding_matrix[2])


# ── Normalization ────────────────────────────────────────────────────


class TestRMSNorm:
    def test_output_shape(self, sample_input):
        norm = RMSNorm(D_MODEL)
        output = norm(sample_input)
        assert output.shape == sample_input.shape

    def test_normalization(self, sample_input):
        norm = RMSNorm(D_MODEL)
        output = norm(sample_input)
        # After RMS normalization (with g=1), the RMS of each vector should be ~1
        rms = torch.sqrt(torch.mean(output ** 2, dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

    def test_preserves_dtype_fp16(self, sample_input):
        norm = RMSNorm(D_MODEL)
        x_fp16 = sample_input.half()
        output = norm(x_fp16)
        assert output.dtype == torch.float16


# ── Feed-Forward ─────────────────────────────────────────────────────


class TestSwiGLUFeedForward:
    def test_output_shape(self, sample_input):
        ffn = SwiGLUFeedForward(D_MODEL)
        ffn.apply(param_init)
        output = ffn(sample_input)
        assert output.shape == sample_input.shape

    def test_hidden_dim_multiple_of_64(self):
        ffn = SwiGLUFeedForward(D_MODEL)
        assert ffn.d_ff % 64 == 0


# ── Rotary Positional Embedding ──────────────────────────────────────


class TestRoPE:
    def test_output_shape(self, device):
        d_k = D_MODEL // NUM_HEADS
        rope = RotaryPositionalEmbedding(10000.0, d_k, 128, device)
        x = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, d_k, device=device)
        positions = torch.arange(SEQ_LEN, device=device).unsqueeze(0)
        output = rope(x, positions)
        assert output.shape == x.shape

    def test_different_positions_differ(self, device):
        d_k = D_MODEL // NUM_HEADS
        rope = RotaryPositionalEmbedding(10000.0, d_k, 128, device)
        x = torch.ones(1, 1, 2, d_k, device=device)
        pos = torch.tensor([0, 5], device=device)
        output = rope(x, pos)
        # Different positions should produce different outputs
        assert not torch.allclose(output[0, 0, 0], output[0, 0, 1])


# ── Softmax ──────────────────────────────────────────────────────────


class TestSoftmax:
    def test_sums_to_one(self):
        x = torch.randn(3, 10)
        s = softmax(x, dim=-1)
        sums = s.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(3), atol=1e-5)

    def test_numerical_stability(self):
        # Very large values should not cause overflow
        x = torch.tensor([[1000.0, 1001.0, 1002.0]])
        s = softmax(x, dim=-1)
        assert not torch.any(torch.isnan(s))
        assert not torch.any(torch.isinf(s))


# ── Attention ────────────────────────────────────────────────────────


class TestMultiHeadAttention:
    def test_self_attention_shape(self, sample_input):
        attn = MultiHeadAttention(D_MODEL, NUM_HEADS, dropout=0.0)
        attn.apply(param_init)
        output = attn(sample_input)
        assert output.shape == sample_input.shape

    def test_cross_attention_shape(self, sample_input):
        attn = MultiHeadAttention(D_MODEL, NUM_HEADS, is_cross=True, dropout=0.0)
        attn.apply(param_init)
        context = torch.randn(BATCH_SIZE, SEQ_LEN * 2, D_MODEL)
        output = attn(sample_input, context=context)
        assert output.shape == sample_input.shape


# ── Causal Mask ──────────────────────────────────────────────────────


class TestCausalMask:
    def test_shape(self):
        mask = causal_mask(5)
        assert mask.shape == (1, 5, 5)

    def test_lower_triangular(self):
        mask = causal_mask(4)
        # First row: only position 0 is allowed
        assert mask[0, 0, 0] == True
        assert mask[0, 0, 1] == False
        # Last row: all positions allowed
        assert mask[0, 3, 0] == True
        assert mask[0, 3, 3] == True


# ── Transformer Blocks ───────────────────────────────────────────────


class TestEncoderBlock:
    def test_output_shape(self, sample_input):
        block = EncoderBlock(D_MODEL, NUM_HEADS, dropout=0.0)
        block.apply(param_init)
        output = block(sample_input)
        assert output.shape == sample_input.shape


class TestDecoderBlock:
    def test_output_shape(self, sample_input):
        block = DecoderBlock(D_MODEL, NUM_HEADS, dropout=0.0)
        block.apply(param_init)
        encoder_output = torch.randn_like(sample_input)
        output = block(sample_input, encoder_output)
        assert output.shape == sample_input.shape


# ── Full Transformer ─────────────────────────────────────────────────


class TestTransformer:
    def test_full_forward_pass(self, sample_ids):
        model = build_transformer(
            src_vocab_size=VOCAB_SIZE,
            tgt_vocab_size=VOCAB_SIZE,
            src_seq_len=SEQ_LEN,
            tgt_seq_len=SEQ_LEN,
            d_model=D_MODEL,
            N=2,
            h=NUM_HEADS,
            dropout=0.0,
        )

        src_mask = torch.ones(BATCH_SIZE, 1, 1, SEQ_LEN).int()
        tgt_mask = causal_mask(SEQ_LEN).expand(BATCH_SIZE, -1, -1, -1).int()

        encoder_output = model.encode(sample_ids, src_mask)
        decoder_output = model.decode(encoder_output, src_mask, sample_ids, tgt_mask)
        logits = model.project(decoder_output)

        assert encoder_output.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)
        assert decoder_output.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)
        assert logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)

    def test_weight_tying(self):
        model = build_transformer(
            src_vocab_size=VOCAB_SIZE,
            tgt_vocab_size=VOCAB_SIZE,
            src_seq_len=SEQ_LEN,
            tgt_seq_len=SEQ_LEN,
            d_model=D_MODEL,
            N=2,
            h=NUM_HEADS,
        )

        # Projection weight should be the same object as target embedding
        assert model.projection_layer.W is model.tgt_embed.embedding_matrix

    def test_param_count_nonzero(self):
        model = build_transformer(
            src_vocab_size=VOCAB_SIZE,
            tgt_vocab_size=VOCAB_SIZE,
            src_seq_len=SEQ_LEN,
            tgt_seq_len=SEQ_LEN,
            d_model=D_MODEL,
            N=2,
            h=NUM_HEADS,
        )
        total = sum(p.numel() for p in model.parameters())
        assert total > 0
