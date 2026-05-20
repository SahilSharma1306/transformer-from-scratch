import torch
import torch.nn as nn
import math


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

    def forward(self, x):
        return x @ self.W.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding_matrix = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )

    def forward(self, token_ids):
        return self.embedding_matrix[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, 1, d_model, device=device, dtype=dtype))

    def forward(self, x):
        in_dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms * self.g).to(in_dtype)


class Position_FeedForward(nn.Module):
    """SwiGLU feed-forward network (Shazeer 2020).

    Uses gated linear units with Swish activation, matching the design
    used in LLaMA / PaLM. Hidden dimension is rounded up to the nearest
    multiple of 64 for hardware alignment.
    """

    def __init__(self, d_model: int, device=None, dtype=None):
        super().__init__()
        hidden_dim = int((8 / 3) * d_model)
        self.d_ff = ((hidden_dim + 63) // 64) * 64
        self.W1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.W3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.W2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x):
        x_gate = self.W1(x)
        x_gate = x_gate * torch.sigmoid(x_gate)  # Swish / SiLU
        x_value = self.W3(x)
        return self.W2(x_gate * x_value)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al., 2021).

    Encodes absolute position information into query and key vectors
    through rotation, enabling relative position awareness in attention.
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        t = torch.arange(max_seq_len, device=device, dtype=inv_freq.dtype)
        freqs = torch.repeat_interleave(torch.outer(t, inv_freq), 2, dim=-1)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x, token_positions):
        if token_positions.dim() == 2:
            cos = self.cos_cached[token_positions]
            sin = self.sin_cached[token_positions]
        else:
            cos = self.cos_cached[token_positions].unsqueeze(0)
            sin = self.sin_cached[token_positions].unsqueeze(0)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        return (x * cos) + (self._rotate_every_two(x) * sin)

    def _rotate_every_two(self, x):
        x0 = x[..., 0::2]
        x1 = x[..., 1::2]
        return torch.stack((-x1, x0), dim=-1).flatten(-2)


def Softmax(x, dim):
    """Numerically stable softmax (manual implementation)."""
    max_val = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - max_val
    num = torch.exp(x_shifted)
    den = torch.sum(num, dim=dim, keepdim=True)
    return num / den


def scaled_dot_product_attention(Q, K, V, mask=None, dropout=None):
    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_scores = Softmax(scores, dim=-1)
    if dropout is not None:
        attention_scores = dropout(attention_scores)
    return attention_scores @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=4096,
                 rope_theta=10000.0, is_cross=False, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.is_cross = is_cross
        self.Wq = Linear(d_model, d_model)
        self.Wk = Linear(d_model, d_model)
        self.Wv = Linear(d_model, d_model)
        self.Wo = Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        if not is_cross:
            self.rope = RotaryPositionalEmbedding(rope_theta, self.d_k, max_seq_len)

    def forward(self, x, context=None, mask=None):
        batch_size, seq_len, _ = x.shape
        kv_input = context if self.is_cross else x
        Q = self.Wq(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.Wk(kv_input).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.Wv(kv_input).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        if not self.is_cross:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            Q = self.rope(Q, positions)
            K = self.rope(K, positions)
        out = scaled_dot_product_attention(Q, K, V, mask=mask, dropout=self.dropout)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.Wo(out)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = Position_FeedForward(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x = x + self.dropout(self.self_attn(self.norm1(x), mask=src_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.norm2 = RMSNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, is_cross=True, dropout=dropout)
        self.norm3 = RMSNorm(d_model)
        self.ffn = Position_FeedForward(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = x + self.dropout(self.self_attn(self.norm1(x), mask=tgt_mask))
        x = x + self.dropout(self.cross_attn(self.norm2(x), context=encoder_output, mask=src_mask))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.norm = RMSNorm(layers[0].norm1.d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.norm = RMSNorm(layers[0].norm1.d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, projection_layer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.projection_layer = projection_layer

    def forward(self, src, src_mask, tgt, tgt_mask):
        return self.project(self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask))

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


def param_init(m):
    """Truncated normal initialization (Glorot-aware)."""
    if isinstance(m, Linear):
        std = (2 / (m.W.shape[0] + m.W.shape[1])) ** 0.5
        nn.init.trunc_normal_(m.W, 0.0, std, -3 * std, 3 * std)
    elif isinstance(m, Embedding):
        nn.init.trunc_normal_(m.embedding_matrix, 0.0, 1.0, -3.0, 3.0)
    elif isinstance(m, RMSNorm):
        nn.init.ones_(m.g)


def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len,
                      d_model=512, N=6, h=8, dropout=0.1, d_ff=2048):
    src_embed = Embedding(src_vocab_size, d_model)
    tgt_embed = Embedding(tgt_vocab_size, d_model)
    encoder = Encoder(nn.ModuleList([EncoderBlock(d_model, h, dropout) for _ in range(N)]))
    decoder = Decoder(nn.ModuleList([DecoderBlock(d_model, h, dropout) for _ in range(N)]))
    projection_layer = Linear(d_model, tgt_vocab_size)
    # Weight tying: saves ~10M params, improves generalisation
    projection_layer.W = tgt_embed.embedding_matrix
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, projection_layer)
    transformer.apply(param_init)
    return transformer


def get_model(config, vocab_src_len, vocab_tgt_len):
    return build_transformer(
        vocab_src_len, vocab_tgt_len,
        config["seq_len"], config["seq_len"],
        d_model=config["d_model"],
        dropout=config.get("dropout", 0.1),
    )
