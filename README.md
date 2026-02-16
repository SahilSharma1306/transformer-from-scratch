<div align="center">

# 🔥 Transformer from Scratch

**A complete Transformer implementation built entirely from first principles in PyTorch.**

No `nn.Linear`. No `nn.Embedding`. No `nn.LayerNorm`. Just raw math.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Architecture](#architecture) · [Quick Start](#quick-start) · [Training Results](#training-results) · [Project Structure](#project-structure)

</div>

---

## ✨ What Makes This Special

This isn't a wrapper around `nn.TransformerEncoder`. Every single component is implemented from scratch using only `torch.Tensor` operations and `nn.Parameter`:

| Component | Standard PyTorch | **This Repo** |
|---|---|---|
| Linear Layer | `nn.Linear` | **Custom `Linear`** — bias-free `y = xW^T` |
| Embedding | `nn.Embedding` | **Custom `Embedding`** — raw lookup table |
| Normalization | `nn.LayerNorm` | **RMSNorm** — faster, no mean subtraction |
| Feed-Forward | `ReLU(xW₁)W₂` | **SwiGLU** — gated activation (LLaMA-style) |
| Positional Encoding | Sinusoidal / Learned | **RoPE** — rotary embeddings |
| Training | Single GPU | **DDP + Mixed Precision (FP16)** |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRANSFORMER                              │
│                                                                 │
│  ┌──────────────────────┐    ┌──────────────────────────────┐   │
│  │      ENCODER (×6)    │    │        DECODER (×6)          │   │
│  │                      │    │                              │   │
│  │  ┌────────────────┐  │    │  ┌────────────────────────┐  │   │
│  │  │   RMSNorm      │  │    │  │   RMSNorm              │  │   │
│  │  │   Self-Attn    │  │    │  │   Masked Self-Attn     │  │   │
│  │  │   + RoPE       │──│────│──│   + RoPE               │  │   │
│  │  │   + Dropout    │  │    │  │   + Dropout             │  │   │
│  │  │   + Residual   │  │    │  │   + Residual            │  │   │
│  │  └────────────────┘  │    │  └────────────────────────┘  │   │
│  │  ┌────────────────┐  │    │  ┌────────────────────────┐  │   │
│  │  │   RMSNorm      │  │    │  │   RMSNorm              │  │   │
│  │  │   SwiGLU FFN   │  │    │  │   Cross-Attention      │  │   │
│  │  │   + Dropout    │  │    │  │   + Dropout             │  │   │
│  │  │   + Residual   │  │    │  │   + Residual            │  │   │
│  │  └────────────────┘  │    │  └────────────────────────┘  │   │
│  │                      │    │  ┌────────────────────────┐  │   │
│  │                      │    │  │   RMSNorm              │  │   │
│  │                      │    │  │   SwiGLU FFN           │  │   │
│  │                      │    │  │   + Dropout             │  │   │
│  │                      │    │  │   + Residual            │  │   │
│  │                      │    │  └────────────────────────┘  │   │
│  └──────────────────────┘    └──────────────────────────────┘   │
│                                            │                    │
│                                     ┌──────▼──────┐            │
│                                     │  Projection  │            │
│                                     │ (weight-tied)│            │
│                                     └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### Mathematical Foundations

<details>
<summary><b>🔢 RMSNorm</b> — Root Mean Square Normalization</summary>

```
RMS(x) = √(mean(x²) + ε)
output = (x / RMS(x)) · γ
```

Unlike LayerNorm, RMSNorm skips the mean subtraction step, making it simpler and ~10% faster while achieving comparable performance.

**Reference:** Zhang & Sennrich, "[Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)" (2019)
</details>

<details>
<summary><b>🔄 RoPE</b> — Rotary Positional Embeddings</summary>

```
For each dimension pair (2i, 2i+1):
    θᵢ = 1 / (10000^(2i/d))

    ┌        ┐   ┌              ┐ ┌        ┐
    │ x'₂ᵢ   │ = │ cos(mθᵢ)  -sin(mθᵢ) │ │ x₂ᵢ   │
    │ x'₂ᵢ₊₁ │   │ sin(mθᵢ)   cos(mθᵢ) │ │ x₂ᵢ₊₁ │
    └        ┘   └              ┘ └        ┘
```

RoPE encodes position by rotating Q/K vectors, making attention scores depend only on relative distances.

**Reference:** Su et al., "[RoFormer](https://arxiv.org/abs/2104.09864)" (2021)
</details>

<details>
<summary><b>⚡ SwiGLU</b> — Gated Feed-Forward Network</summary>

```
gate   = SiLU(x · W₁ᵀ)        where SiLU(z) = z · σ(z)
value  = x · W₃ᵀ
output = (gate ⊙ value) · W₂ᵀ
```

SwiGLU replaces the standard ReLU FFN with a gated mechanism, improving training efficiency. Hidden dim follows LLaMA: `ceil(8/3 × d_model / 64) × 64`.

**Reference:** Shazeer, "[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)" (2020)
</details>

<details>
<summary><b>🎯 Scaled Dot-Product Attention</b></summary>

```
Attention(Q, K, V) = softmax(Q · Kᵀ / √d_k) · V
```

Multi-head attention splits Q, K, V into `h` heads, applies attention independently, and concatenates results.

**Reference:** Vaswani et al., "[Attention Is All You Need](https://arxiv.org/abs/1706.03762)" (2017)
</details>

---

## Quick Start

### Installation

```bash
git clone https://github.com/SahilSharma1306/transformer-from-scratch.git
cd transformer-from-scratch
pip install -r requirements.txt
```

### Training

```bash
# Single GPU
python -m transformer.train --config configs/default.yaml

# Multi-GPU (DDP) — auto-detects available GPUs
bash scripts/train.sh

# Custom configuration
python -m transformer.train --config configs/default.yaml --num_epochs 50 --batch_size 64
```

### Translation (after training)

```bash
# Single sentence
python -m transformer.translate --checkpoint weights/tmodel_24.pt --text "Hello world"

# Interactive mode
python -m transformer.translate --checkpoint weights/tmodel_24.pt --interactive
```

### Run Tests

```bash
python -m pytest tests/ -v
```

---

## Training Results

Trained on the **Opus Books** English → Italian dataset (32,332 sentence pairs) using 2× T4 GPUs on Kaggle.

### Configuration

| Hyperparameter | Value |
|---|---|
| Model Dimension (`d_model`) | 512 |
| Attention Heads | 8 |
| Encoder/Decoder Layers | 6 |
| SwiGLU Hidden Dim | 1,408 |
| Vocabulary Size | 20,000 (BPE) |
| Sequence Length | 320 |
| Batch Size | 32 per GPU |
| Peak Learning Rate | 6×10⁻⁴ |
| Dropout | 0.3 |
| Training | 25 epochs, DDP, FP16 |

### Loss Progression

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 0 | ~7.5 | 6.14 |
| 5 | ~5.3 | 5.02 |
| 10 | ~4.6 | 4.65 |
| 15 | ~3.9 | 4.51 |
| 20 | ~3.7 | 4.47 |
| 24 | ~3.6 | 4.48 |

### Sample Translations (Epoch 24)

| Source (EN) | Target (IT) | Prediction |
|---|---|---|
| "But Oblonsky arranged that too." | "Stepan Arkad'ic accomodò anche questo." | "Ma Stepan Arkad'ic aveva preso questo." |
| "Karenin glanced at him with his weary eyes." | "Aleksej Aleksandrovic lo guardò con occhi stanchi." | "Aleksej Aleksandrovic lo guardò con gli occhi." |
| "'What do you mean?" | "— Che cosa allora?" | "— Che cosa volete?" |

---

## Project Structure

```
transformer-from-scratch/
│
├── transformer/                 # Core package
│   ├── __init__.py              # Package exports
│   ├── config.py                # Dataclass config + CLI/YAML parsing
│   ├── model.py                 # All model components (from scratch)
│   ├── dataset.py               # Data loading + BPE tokenization
│   ├── train.py                 # DDP + AMP training loop
│   ├── validate.py              # Validation + greedy decoding
│   └── translate.py             # Standalone inference script
│
├── tests/
│   └── test_model.py            # Unit tests for all components
│
├── configs/
│   └── default.yaml             # Default hyperparameters
│
├── scripts/
│   └── train.sh                 # DDP launch script
│
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE                      # MIT
├── Makefile                     # Common operations
├── README.md
├── requirements.txt
└── pyproject.toml               # Python packaging
```

---

## Configuration

All hyperparameters live in a `TransformerConfig` dataclass. Override via:

**YAML file:**
```yaml
# configs/custom.yaml
num_epochs: 50
batch_size: 64
lr: 3e-4
```

**CLI flags** (override YAML):
```bash
python -m transformer.train --config configs/custom.yaml --dropout 0.2
```

**Python:**
```python
from transformer import TransformerConfig
config = TransformerConfig(d_model=768, num_heads=12, num_layers=12)
```

---

## Implementation Details

### Design Decisions

1. **Pre-Norm Residuals** — We normalize *before* attention/FFN, not after. This improves training stability and removes the need for learning rate warmup tuning.

2. **Weight Tying** — The output projection layer shares its weight matrix with the target embedding, reducing parameters and improving generalization.

3. **No Bias** — All `Linear` layers are bias-free, following modern LLM conventions (LLaMA, GPT-NeoX).

4. **DDP-Safe Tokenizer** — Only Rank 0 builds the BPE tokenizer; other ranks wait at a `dist.barrier()` before loading from disk.

5. **Truncated Normal Init** — Xavier-style initialization with ±3σ truncation for stable training from the start.

---

## References

- Vaswani et al., "[Attention Is All You Need](https://arxiv.org/abs/1706.03762)" (2017)
- Su et al., "[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)" (2021)
- Zhang & Sennrich, "[Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)" (2019)
- Shazeer, "[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)" (2020)
- Touvron et al., "[LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)" (2023)

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
<b>Built with ❤️ and raw tensors</b>
</div>
