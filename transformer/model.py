"""
Transformer Model — Built Entirely from Scratch
================================================

Every component in this file is implemented from first principles using
only ``torch.Tensor`` operations and ``nn.Parameter``.  No ``nn.Linear``,
no ``nn.Embedding``, no ``nn.LayerNorm`` — just raw math.

Architecture Highlights
-----------------------
- **RMSNorm** instead of LayerNorm (faster, no mean subtraction)
- **SwiGLU** feed-forward networks (LLaMA-style gated activation)
- **Rotary Positional Embeddings** (RoPE) — no sinusoidal/learned PE
- **Weight Tying** between target embedding and output projection
- **Pre-Norm** residual connections (norm before attention/FFN)

References
----------
- Vaswani et al., "Attention Is All You Need" (2017)
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
- Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019)
- Shazeer, "GLU Variants Improve Transformer" (2020)
- Touvron et al., "LLaMA: Open and Efficient Foundation Language Models" (2023)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════
#  Primitive Layers
# ═══════════════════════════════════════════════════════════════════════


class Linear(nn.Module):
    """Bias-free linear projection: y = xW^T.

    Unlike ``nn.Linear``, this layer has **no bias** and stores only
    a single weight matrix.  This makes it compatible with weight tying
    (sharing parameters between embedding and output projection).

    Parameters
    ----------
    in_features : int
        Size of the input dimension.
    out_features : int
        Size of the output dimension.
    device : torch.device, optional
        Device to place parameters on.
    dtype : torch.dtype, optional
        Data type for parameters.

    Shape
    -----
    - Input:  ``(*, in_features)``
    - Output: ``(*, out_features)``
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.W = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: ``y = x @ W^T``.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(*, in_features)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(*, out_features)``.
        """
        return x @ self.W.T


class Embedding(nn.Module):
    """Lookup-table embedding: maps integer token IDs to dense vectors.

    Built from a raw ``nn.Parameter`` matrix instead of using
    ``nn.Embedding``, to demonstrate the underlying mechanism.

    Parameters
    ----------
    num_embeddings : int
        Size of the vocabulary (number of rows).
    embedding_dim : int
        Dimensionality of each embedding vector (number of columns).
    device : torch.device, optional
        Device to place parameters on.
    dtype : torch.dtype, optional
        Data type for parameters.

    Shape
    -----
    - Input:  ``(B, T)`` — batch of integer token IDs
    - Output: ``(B, T, embedding_dim)``
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding_matrix = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up embeddings for the given token IDs.

        Parameters
        ----------
        token_ids : torch.Tensor
            Integer tensor of shape ``(B, T)``.

        Returns
        -------
        torch.Tensor
            Embedded representations of shape ``(B, T, D)``.
        """
        return self.embedding_matrix[token_ids]


# ═══════════════════════════════════════════════════════════════════════
#  Normalization
# ═══════════════════════════════════════════════════════════════════════


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Unlike standard LayerNorm, RMSNorm skips the mean-subtraction step
    and normalizes only by the root-mean-square of activations.  This
    is both simpler and faster while achieving comparable performance.

    Formula::

        RMS(x) = sqrt( mean(x^2) + eps )
        output = (x / RMS(x)) * g

    where ``g`` is a learnable gain parameter initialized to ones.

    References
    ----------
    Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019)

    Parameters
    ----------
    d_model : int
        Dimensionality of the input features.
    eps : float
        Small constant for numerical stability.
    device : torch.device, optional
        Device to place parameters on.
    dtype : torch.dtype, optional
        Data type for parameters.

    Shape
    -----
    - Input:  ``(B, T, D)``
    - Output: ``(B, T, D)``
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, 1, d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Computation is done in float32 for numerical stability,
        then cast back to the input dtype (important for mixed precision).

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, T, D)``.

        Returns
        -------
        torch.Tensor
            Normalized output of shape ``(B, T, D)``.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # RMS = sqrt( mean(x_i^2) + eps )
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

        # Normalize and scale
        x = (x / rms) * self.g
        return x.to(in_dtype)


# ═══════════════════════════════════════════════════════════════════════
#  Feed-Forward Network
# ═══════════════════════════════════════════════════════════════════════


class SwiGLUFeedForward(nn.Module):
    """SwiGLU-gated Feed-Forward Network (LLaMA-style).

    Instead of the standard ``ReLU(xW1) @ W2`` FFN from the original
    Transformer, this uses a gated linear unit with SiLU activation::

        gate   = SiLU(x @ W1^T)      # Gating branch
        value  = x @ W3^T            # Value branch
        hidden = gate * value         # Element-wise gating
        output = hidden @ W2^T       # Down-projection

    The hidden dimension is set to ``ceil(8/3 * d_model / 64) * 64``
    following LLaMA conventions (rounded up to the nearest multiple
    of 64 for GPU efficiency).

    References
    ----------
    Shazeer, "GLU Variants Improve Transformer" (2020)
    Touvron et al., "LLaMA" (2023)

    Parameters
    ----------
    d_model : int
        Input and output dimensionality.
    device : torch.device, optional
        Device to place parameters on.
    dtype : torch.dtype, optional
        Data type for parameters.

    Shape
    -----
    - Input:  ``(B, T, D)``
    - Output: ``(B, T, D)``
    """

    def __init__(
        self,
        d_model: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        # LLaMA-style hidden dim: 8/3 * d_model, rounded to multiple of 64
        hidden_dim = int((8 / 3) * d_model)
        self.d_ff = ((hidden_dim + 63) // 64) * 64

        self.W1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)  # Gate projection
        self.W3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)  # Value projection
        self.W2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)  # Down projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU feed-forward transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, T, D)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(B, T, D)``.
        """
        # Gate branch: SiLU(x @ W1^T)  where SiLU(z) = z * sigmoid(z)
        x_gate = self.W1(x)                          # (B, T, d_ff)
        x_gate = x_gate * torch.sigmoid(x_gate)      # SiLU activation

        # Value branch
        x_value = self.W3(x)                          # (B, T, d_ff)

        # Gated combination + down-projection
        x_hidden = x_gate * x_value                   # (B, T, d_ff)
        return self.W2(x_hidden)                      # (B, T, D)


# ═══════════════════════════════════════════════════════════════════════
#  Positional Encoding — Rotary Positional Embeddings (RoPE)
# ═══════════════════════════════════════════════════════════════════════


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE).

    RoPE encodes position information by rotating the query and key
    vectors in 2D subspaces.  Unlike additive sinusoidal embeddings,
    RoPE makes the dot product between q and k depend only on their
    relative distance, enabling better length generalization.

    Formula::

        For each pair of dimensions (2i, 2i+1):
            freq_i    = 1 / (theta^(2i/d))
            cos_m     = cos(m * freq_i)
            sin_m     = sin(m * freq_i)

            [x_2i  ]     [cos_m  -sin_m] [x_2i  ]
            [x_2i+1]  =  [sin_m   cos_m] [x_2i+1]

    References
    ----------
    Su et al., "RoFormer: Enhanced Transformer with Rotary Position
    Embedding" (2021)

    Parameters
    ----------
    theta : float
        Base frequency (10000.0 in the original paper).
    d_k : int
        Dimension per attention head.
    max_seq_len : int
        Maximum sequence length to precompute frequencies for.
    device : torch.device, optional
        Device to place buffers on.

    Shape
    -----
    - Input:  ``(B, H, T, d_k)``
    - Output: ``(B, H, T, d_k)``
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        # Inverse frequencies: 1 / (theta^(2i/d))  for i = 0, 1, ..., d_k/2 - 1
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))

        # Position indices: 0, 1, 2, ..., max_seq_len - 1
        t = torch.arange(max_seq_len, device=device, dtype=inv_freq.dtype)

        # Outer product: (max_seq_len, d_k/2)
        freqs = torch.outer(t, inv_freq)

        # Duplicate each frequency for paired dimensions: (max_seq_len, d_k)
        freqs = torch.repeat_interleave(freqs, 2, dim=-1)

        # Cache cos and sin values (not learnable)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, H, T, d_k)``.
        token_positions : torch.Tensor
            Position indices of shape ``(B, T)`` or ``(T,)``.

        Returns
        -------
        torch.Tensor
            Rotated tensor of shape ``(B, H, T, d_k)``.
        """
        if token_positions.dim() == 2:
            cos = self.cos_cached[token_positions]     # (B, T, d_k)
            sin = self.sin_cached[token_positions]
        else:
            cos = self.cos_cached[token_positions].unsqueeze(0)  # (1, T, d_k)
            sin = self.sin_cached[token_positions].unsqueeze(0)

        cos = cos.unsqueeze(1)  # (B, 1, T, d_k) — broadcast over heads
        sin = sin.unsqueeze(1)

        return (x * cos) + (self._rotate_every_two(x) * sin)

    @staticmethod
    def _rotate_every_two(x: torch.Tensor) -> torch.Tensor:
        """Rotate adjacent pairs: [x0, x1, x2, x3, ...] → [-x1, x0, -x3, x2, ...].

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(..., d_k)`` where ``d_k`` is even.

        Returns
        -------
        torch.Tensor
            Rotated tensor of the same shape.
        """
        x0 = x[..., 0::2]   # Even indices
        x1 = x[..., 1::2]   # Odd indices
        x_rotated = torch.stack((-x1, x0), dim=-1)
        return x_rotated.flatten(-2)


# ═══════════════════════════════════════════════════════════════════════
#  Attention
# ═══════════════════════════════════════════════════════════════════════


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Numerically stable softmax (from scratch).

    Subtracts the max value before exponentiation to prevent overflow::

        softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

    Parameters
    ----------
    x : torch.Tensor
        Input logits.
    dim : int
        Dimension along which to compute softmax.

    Returns
    -------
    torch.Tensor
        Probability distribution (same shape as input).
    """
    max_val = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - max_val
    num = torch.exp(x_shifted)
    den = torch.sum(num, dim=dim, keepdim=True)
    return num / den


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> torch.Tensor:
    """Scaled Dot-Product Attention.

    Computes::

        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Parameters
    ----------
    Q : torch.Tensor
        Queries of shape ``(B, H, T_q, d_k)``.
    K : torch.Tensor
        Keys of shape ``(B, H, T_k, d_k)``.
    V : torch.Tensor
        Values of shape ``(B, H, T_k, d_k)``.
    mask : torch.Tensor, optional
        Boolean mask of shape ``(B, 1, T_q, T_k)`` or broadcastable.
        Positions with ``0`` are filled with ``-inf`` before softmax.
    dropout : nn.Dropout, optional
        Dropout applied to attention weights.

    Returns
    -------
    torch.Tensor
        Attention output of shape ``(B, H, T_q, d_k)``.
    """
    d_k = Q.shape[-1]

    # (B, H, T_q, d_k) @ (B, H, d_k, T_k) → (B, H, T_q, T_k)
    scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attention_weights = softmax(scores, dim=-1)

    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # (B, H, T_q, T_k) @ (B, H, T_k, d_k) → (B, H, T_q, d_k)
    return attention_weights @ V


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with Rotary Positional Embeddings.

    Supports both **self-attention** (queries, keys, values from same
    sequence) and **cross-attention** (queries from decoder, keys/values
    from encoder).  RoPE is only applied to self-attention since
    cross-attention keys come from a different sequence.

    Formula::

        MultiHead(x) = Concat(head_1, ..., head_h) @ W_o

        where head_i = Attention(x @ W_q_i, kv @ W_k_i, kv @ W_v_i)

    Parameters
    ----------
    d_model : int
        Total model dimensionality.
    num_heads : int
        Number of parallel attention heads.
    max_seq_len : int
        Maximum sequence length for RoPE precomputation.
    rope_theta : float
        Base frequency for RoPE.
    is_cross : bool
        If True, this is a cross-attention layer (no RoPE).
    dropout : float
        Dropout probability on attention weights.

    Shape
    -----
    - Input:  ``x`` of shape ``(B, T, D)``, optional ``context`` of shape ``(B, T_enc, D)``
    - Output: ``(B, T, D)``
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 4096,
        rope_theta: float = 10000.0,
        is_cross: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.is_cross = is_cross

        # Projection matrices (all bias-free)
        self.Wq = Linear(d_model, d_model)
        self.Wk = Linear(d_model, d_model)
        self.Wv = Linear(d_model, d_model)
        self.Wo = Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # RoPE only for self-attention (cross-attention keys are from a different sequence)
        if not is_cross:
            self.rope = RotaryPositionalEmbedding(rope_theta, self.d_k, max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute multi-head attention.

        Parameters
        ----------
        x : torch.Tensor
            Query input of shape ``(B, T, D)``.
        context : torch.Tensor, optional
            Key/value source for cross-attention ``(B, T_enc, D)``.
            If None, uses ``x`` (self-attention).
        mask : torch.Tensor, optional
            Attention mask.

        Returns
        -------
        torch.Tensor
            Attention output of shape ``(B, T, D)``.
        """
        batch_size, seq_len, _ = x.shape

        # Project queries from x, keys/values from context (or x for self-attn)
        Q = self.Wq(x)                                            # (B, T, D)
        kv_input = context if self.is_cross else x
        K = self.Wk(kv_input)                                     # (B, T_kv, D)
        V = self.Wv(kv_input)                                     # (B, T_kv, D)

        # Reshape to multi-head format: (B, T, D) → (B, H, T, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply RoPE to queries and keys (self-attention only)
        if not self.is_cross:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            Q = self.rope(Q, positions)
            K = self.rope(K, positions)

        # Scaled dot-product attention
        attn_output = scaled_dot_product_attention(Q, K, V, mask=mask, dropout=self.dropout)

        # Concatenate heads: (B, H, T, d_k) → (B, T, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Final linear projection
        return self.Wo(attn_output)                                # (B, T, D)


# ═══════════════════════════════════════════════════════════════════════
#  Parameter Initialization
# ═══════════════════════════════════════════════════════════════════════


def param_init(m: nn.Module) -> None:
    """Initialize model parameters using truncated normal distribution.

    Initialization strategy:
    - **Linear layers**: Truncated normal with std = sqrt(2 / (fan_in + fan_out))
      (Xavier-style, but truncated at ±3σ for stability)
    - **Embedding layers**: Truncated normal N(0, 1), clipped at ±3
    - **RMSNorm**: Gain initialized to ones

    Parameters
    ----------
    m : nn.Module
        Module to initialize (called via ``model.apply(param_init)``).
    """
    if isinstance(m, Linear):
        fan_in, fan_out = m.W.shape[1], m.W.shape[0]
        std = (2.0 / (fan_in + fan_out)) ** 0.5
        torch.nn.init.trunc_normal_(m.W, 0.0, std, -3 * std, 3 * std)
    elif isinstance(m, Embedding):
        torch.nn.init.trunc_normal_(m.embedding_matrix, 0.0, 1.0, -3.0, 3.0)
    elif isinstance(m, RMSNorm):
        torch.nn.init.ones_(m.g)


# ═══════════════════════════════════════════════════════════════════════
#  Transformer Blocks
# ═══════════════════════════════════════════════════════════════════════


class EncoderBlock(nn.Module):
    """Single Transformer Encoder Block.

    Architecture (Pre-Norm Residual)::

        x ──→ RMSNorm ──→ Self-Attention ──→ Dropout ──→ (+) ──→
        │                                                  ↑
        └──────────────────────────────────────────────────┘
              ──→ RMSNorm ──→ SwiGLU FFN ──→ Dropout ──→ (+) ──→
              │                                            ↑
              └────────────────────────────────────────────┘

    Parameters
    ----------
    d_model : int
        Model dimensionality.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFeedForward(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with pre-norm residual connections.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, T, D)``.
        src_mask : torch.Tensor, optional
            Source padding mask.

        Returns
        -------
        torch.Tensor
            Output of shape ``(B, T, D)``.
        """
        # Sub-layer 1: Self-Attention
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, mask=src_mask)
        x = residual + self.dropout(x)

        # Sub-layer 2: Feed-Forward
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)

        return x


class DecoderBlock(nn.Module):
    """Single Transformer Decoder Block.

    Architecture (Pre-Norm Residual)::

        x ──→ RMSNorm ──→ Masked Self-Attention ──→ Dropout ──→ (+) ──→
              ──→ RMSNorm ──→ Cross-Attention ──→ Dropout ──→ (+) ──→
              ──→ RMSNorm ──→ SwiGLU FFN ──→ Dropout ──→ (+) ──→

    Parameters
    ----------
    d_model : int
        Model dimensionality.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)

        self.norm2 = RMSNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, is_cross=True, dropout=dropout)

        self.norm3 = RMSNorm(d_model)
        self.ffn = SwiGLUFeedForward(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with three sub-layers.

        Parameters
        ----------
        x : torch.Tensor
            Decoder input of shape ``(B, T_tgt, D)``.
        encoder_output : torch.Tensor
            Encoder output of shape ``(B, T_src, D)``.
        src_mask : torch.Tensor, optional
            Source padding mask.
        tgt_mask : torch.Tensor, optional
            Target causal + padding mask.

        Returns
        -------
        torch.Tensor
            Output of shape ``(B, T_tgt, D)``.
        """
        # Sub-layer 1: Masked Self-Attention
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, mask=tgt_mask)
        x = residual + self.dropout(x)

        # Sub-layer 2: Cross-Attention (attending to encoder output)
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, context=encoder_output, mask=src_mask)
        x = residual + self.dropout(x)

        # Sub-layer 3: Feed-Forward
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)

        return x


# ═══════════════════════════════════════════════════════════════════════
#  Encoder & Decoder Stacks
# ═══════════════════════════════════════════════════════════════════════


class Encoder(nn.Module):
    """Stack of N Encoder blocks with a final RMSNorm.

    Parameters
    ----------
    layers : nn.ModuleList
        List of ``EncoderBlock`` instances.
    """

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = RMSNorm(layers[0].norm1.d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pass input through all encoder blocks.

        Parameters
        ----------
        x : torch.Tensor
            Embedded source tokens ``(B, T, D)``.
        mask : torch.Tensor, optional
            Source padding mask.

        Returns
        -------
        torch.Tensor
            Encoded representations ``(B, T, D)``.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N Decoder blocks with a final RMSNorm.

    Parameters
    ----------
    layers : nn.ModuleList
        List of ``DecoderBlock`` instances.
    """

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = RMSNorm(layers[0].norm1.d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pass input through all decoder blocks.

        Parameters
        ----------
        x : torch.Tensor
            Embedded target tokens ``(B, T_tgt, D)``.
        encoder_output : torch.Tensor
            Encoder output ``(B, T_src, D)``.
        src_mask : torch.Tensor, optional
            Source padding mask.
        tgt_mask : torch.Tensor, optional
            Target causal mask.

        Returns
        -------
        torch.Tensor
            Decoded representations ``(B, T_tgt, D)``.
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


# ═══════════════════════════════════════════════════════════════════════
#  Full Transformer
# ═══════════════════════════════════════════════════════════════════════


class Transformer(nn.Module):
    """Complete Encoder-Decoder Transformer for sequence-to-sequence tasks.

    This model follows the architecture from "Attention Is All You Need"
    with modern improvements:

    - Pre-norm residual connections (norm before attention/FFN)
    - RMSNorm instead of LayerNorm
    - SwiGLU activation instead of ReLU
    - Rotary Positional Embeddings instead of sinusoidal
    - Weight tying between target embedding and output projection

    Parameters
    ----------
    encoder : Encoder
        The encoder stack.
    decoder : Decoder
        The decoder stack.
    src_embed : Embedding
        Source language embedding layer.
    tgt_embed : Embedding
        Target language embedding layer.
    projection_layer : Linear
        Output projection (vocabulary logits).  Weights are tied with
        ``tgt_embed``.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: Embedding,
        tgt_embed: Embedding,
        projection_layer: Linear,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.projection_layer = projection_layer

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode source sequence.

        Parameters
        ----------
        src : torch.Tensor
            Source token IDs ``(B, T_src)``.
        src_mask : torch.Tensor, optional
            Source padding mask.

        Returns
        -------
        torch.Tensor
            Encoder output ``(B, T_src, D)``.
        """
        src = self.src_embed(src)     # (B, T_src, D)
        return self.encoder(src, src_mask)

    def decode(
        self,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        tgt: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode target sequence given encoder output.

        Parameters
        ----------
        encoder_output : torch.Tensor
            Encoder output ``(B, T_src, D)``.
        src_mask : torch.Tensor, optional
            Source padding mask.
        tgt : torch.Tensor
            Target token IDs ``(B, T_tgt)``.
        tgt_mask : torch.Tensor, optional
            Target causal mask.

        Returns
        -------
        torch.Tensor
            Decoder output ``(B, T_tgt, D)``.
        """
        tgt = self.tgt_embed(tgt)     # (B, T_tgt, D)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project decoder output to vocabulary log-probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Decoder output ``(B, T, D)``.

        Returns
        -------
        torch.Tensor
            Log-probabilities over vocabulary ``(B, T, V)``.
        """
        return torch.log_softmax(self.projection_layer(x), dim=-1)


# ═══════════════════════════════════════════════════════════════════════
#  Model Factory
# ═══════════════════════════════════════════════════════════════════════


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
) -> Transformer:
    """Build a complete Transformer model from configuration.

    Constructs embeddings, encoder stack, decoder stack, and output
    projection.  Applies **weight tying** between the target embedding
    and the output projection layer.

    Parameters
    ----------
    src_vocab_size : int
        Source vocabulary size.
    tgt_vocab_size : int
        Target vocabulary size.
    src_seq_len : int
        Maximum source sequence length.
    tgt_seq_len : int
        Maximum target sequence length.
    d_model : int
        Model dimensionality (default: 512).
    N : int
        Number of encoder/decoder blocks (default: 6).
    h : int
        Number of attention heads (default: 8).
    dropout : float
        Dropout probability (default: 0.1).

    Returns
    -------
    Transformer
        Initialized Transformer model ready for training.
    """
    # Embeddings
    src_embed = Embedding(src_vocab_size, d_model)
    tgt_embed = Embedding(tgt_vocab_size, d_model)

    # Encoder
    encoder_blocks = [EncoderBlock(d_model, h, dropout) for _ in range(N)]
    encoder = Encoder(nn.ModuleList(encoder_blocks))

    # Decoder
    decoder_blocks = [DecoderBlock(d_model, h, dropout) for _ in range(N)]
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Output projection (weight-tied with target embedding)
    projection_layer = Linear(d_model, tgt_vocab_size)
    projection_layer.W = tgt_embed.embedding_matrix  # Weight tying!

    # Assemble and initialize
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, projection_layer)
    transformer.apply(param_init)

    return transformer


def get_model(config, vocab_src_len: int, vocab_tgt_len: int) -> Transformer:
    """Convenience factory that builds a Transformer from a config object.

    Parameters
    ----------
    config : TransformerConfig or dict
        Configuration containing ``seq_len``, ``d_model``, ``dropout``,
        ``num_layers``, and ``num_heads``.
    vocab_src_len : int
        Source vocabulary size.
    vocab_tgt_len : int
        Target vocabulary size.

    Returns
    -------
    Transformer
        Ready-to-train model.
    """
    if hasattr(config, "d_model"):
        # Dataclass config
        return build_transformer(
            vocab_src_len, vocab_tgt_len,
            config.seq_len, config.seq_len,
            config.d_model,
            N=config.num_layers,
            h=config.num_heads,
            dropout=config.dropout,
        )
    else:
        # Dict config (backward compatibility)
        dropout = config.get("dropout", 0.1)
        return build_transformer(
            vocab_src_len, vocab_tgt_len,
            config["seq_len"], config["seq_len"],
            config["d_model"],
            dropout=dropout,
        )


def model_summary(model: Transformer) -> str:
    """Generate a human-readable summary of model parameters.

    Parameters
    ----------
    model : Transformer
        The model to summarize.

    Returns
    -------
    str
        Formatted string with parameter counts per component.
    """
    lines = ["=" * 60, "MODEL SUMMARY", "=" * 60]

    components = {
        "Source Embedding": model.src_embed,
        "Target Embedding": model.tgt_embed,
        "Encoder": model.encoder,
        "Decoder": model.decoder,
        "Projection": model.projection_layer,
    }

    total = 0
    for name, module in components.items():
        params = sum(p.numel() for p in module.parameters())
        total += params
        lines.append(f"  {name:25s} {params:>12,d}")

    # Account for weight tying (projection shares weights with tgt_embed)
    tied = sum(p.numel() for p in model.projection_layer.parameters())
    unique = total - tied

    lines.append("-" * 60)
    lines.append(f"  {'Total (with tying)':25s} {total:>12,d}")
    lines.append(f"  {'Unique (weight-tied)':25s} {unique:>12,d}")
    lines.append("=" * 60)

    return "\n".join(lines)
