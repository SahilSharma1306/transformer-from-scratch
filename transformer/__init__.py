"""
Transformer from Scratch
========================

A complete Transformer implementation built entirely from scratch in PyTorch.
No nn.Linear, no nn.Embedding — every component hand-written for deep understanding.

Features:
    - RMSNorm (Root Mean Square Normalization)
    - SwiGLU Feed-Forward Networks (LLaMA-style)
    - Rotary Positional Embeddings (RoPE)
    - Multi-Head Self & Cross Attention
    - Distributed Data Parallel (DDP) Training
    - Mixed Precision Training (FP16)
    - Cosine Learning Rate Schedule with Linear Warmup

Author: Sahil Sharma
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Sahil Sharma"

from transformer.model import build_transformer, Transformer
from transformer.config import TransformerConfig

__all__ = ["build_transformer", "Transformer", "TransformerConfig"]
