"""
Dataset & Tokenization Pipeline
================================

Handles data loading, BPE tokenizer training, and batched dataset
construction for the bilingual translation task.

Pipeline::

    Opus Books (HuggingFace) → BPE Tokenizer → BilingualDataset → DataLoader

DDP Safety
----------
All I/O operations (downloading data, building tokenizers) are guarded
so that only Rank 0 performs writes, with ``dist.barrier()`` ensuring
other ranks wait before reading.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterator, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Masking Utilities
# ═══════════════════════════════════════════════════════════════════════


def causal_mask(size: int) -> torch.Tensor:
    """Create a causal (autoregressive) mask.

    The mask allows each position to attend only to itself and
    earlier positions.  Future positions are blocked.

    Parameters
    ----------
    size : int
        Sequence length.

    Returns
    -------
    torch.Tensor
        Boolean mask of shape ``(1, size, size)`` where ``True``
        means "allowed to attend" and ``False`` means "blocked".

    Example
    -------
    For size=4::

        [[True,  False, False, False],
         [True,  True,  False, False],
         [True,  True,  True,  False],
         [True,  True,  True,  True ]]
    """
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0


# ═══════════════════════════════════════════════════════════════════════
#  Bilingual Dataset
# ═══════════════════════════════════════════════════════════════════════


class BilingualDataset(Dataset):
    """PyTorch Dataset for bilingual sentence pairs.

    Each item returns padded encoder input, decoder input, label,
    and corresponding masks — ready for the Transformer.

    Parameters
    ----------
    ds : Dataset
        Raw HuggingFace dataset split.
    tokenizer_src : Tokenizer
        Source language BPE tokenizer.
    tokenizer_tgt : Tokenizer
        Target language BPE tokenizer.
    src_lang : str
        Source language key in the dataset.
    tgt_lang : str
        Target language key in the dataset.
    seq_len : int
        Fixed sequence length (sentences are padded/truncated to this).
    """

    def __init__(
        self,
        ds,
        tokenizer_src: Tokenizer,
        tokenizer_tgt: Tokenizer,
        src_lang: str,
        tgt_lang: str,
        seq_len: int,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # Store special token IDs as plain integers (not tensors)
        # to avoid "nested list of tensors" errors during torch.cat
        self.sos_token_src: int = tokenizer_src.token_to_id("[SOS]")
        self.eos_token_src: int = tokenizer_src.token_to_id("[EOS]")
        self.pad_token_src: int = tokenizer_src.token_to_id("[PAD]")

        self.sos_token_tgt: int = tokenizer_tgt.token_to_id("[SOS]")
        self.eos_token_tgt: int = tokenizer_tgt.token_to_id("[EOS]")
        self.pad_token_tgt: int = tokenizer_tgt.token_to_id("[PAD]")

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> dict:
        """Get a single training example.

        Returns
        -------
        dict
            Keys: ``encoder_input``, ``decoder_input``, ``encoder_mask``,
            ``decoder_mask``, ``label``, ``src_text``, ``tgt_text``.

        Layout
        ------
        - Encoder input: ``[SOS] tokens [EOS] [PAD...]``
        - Decoder input: ``[SOS] tokens [PAD...]``
        - Label:         ``tokens [EOS] [PAD...]``
        """
        src_target_pair = self.ds[index]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        # Tokenize
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Calculate padding needed
        enc_num_padding = self.seq_len - len(enc_input_tokens) - 2  # -2 for [SOS] + [EOS]
        dec_num_padding = self.seq_len - len(dec_input_tokens) - 1  # -1 for [SOS]

        # Truncate if sentence exceeds max length
        if enc_num_padding < 0 or dec_num_padding < 0:
            enc_input_tokens = enc_input_tokens[: self.seq_len - 2]
            dec_input_tokens = dec_input_tokens[: self.seq_len - 1]
            enc_num_padding = self.seq_len - len(enc_input_tokens) - 2
            dec_num_padding = self.seq_len - len(dec_input_tokens) - 1

        # Build padded tensors
        encoder_padding = torch.full((enc_num_padding,), self.pad_token_src, dtype=torch.int64)
        decoder_padding = torch.full((dec_num_padding,), self.pad_token_tgt, dtype=torch.int64)

        # Encoder: [SOS] + tokens + [EOS] + padding
        encoder_input = torch.cat([
            torch.tensor([self.sos_token_src], dtype=torch.int64),
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            torch.tensor([self.eos_token_src], dtype=torch.int64),
            encoder_padding,
        ])

        # Decoder: [SOS] + tokens + padding
        decoder_input = torch.cat([
            torch.tensor([self.sos_token_tgt], dtype=torch.int64),
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            decoder_padding,
        ])

        # Label: tokens + [EOS] + padding  (shifted right by one from decoder input)
        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.eos_token_tgt], dtype=torch.int64),
            decoder_padding,
        ])

        # Shape assertions
        assert encoder_input.size(0) == self.seq_len, f"Encoder input shape mismatch: {encoder_input.size(0)} != {self.seq_len}"
        assert decoder_input.size(0) == self.seq_len, f"Decoder input shape mismatch: {decoder_input.size(0)} != {self.seq_len}"
        assert label.size(0) == self.seq_len, f"Label shape mismatch: {label.size(0)} != {self.seq_len}"

        return {
            "encoder_input": encoder_input,                                                      # (seq_len,)
            "decoder_input": decoder_input,                                                      # (seq_len,)
            "encoder_mask": (encoder_input != self.pad_token_src).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token_tgt).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),  # (1, seq_len, seq_len)
            "label": label,                                                                      # (seq_len,)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


# ═══════════════════════════════════════════════════════════════════════
#  Tokenizer
# ═══════════════════════════════════════════════════════════════════════


def get_all_sentences(ds, lang: str) -> Iterator[str]:
    """Yield all sentences for a given language from the dataset.

    Parameters
    ----------
    ds : Dataset
        HuggingFace dataset with ``translation`` field.
    lang : str
        Language key (e.g. ``"en"``).

    Yields
    ------
    str
        Individual sentences.
    """
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(config, ds, lang: str) -> Tokenizer:
    """Load an existing tokenizer or train a new BPE tokenizer.

    DDP-safe: only Rank 0 builds the tokenizer; other ranks wait
    at a barrier.

    Parameters
    ----------
    config : TransformerConfig or dict
        Must have ``tokenizer_file`` and ``vocab_size`` fields.
    ds : Dataset
        Training data to build tokenizer from.
    lang : str
        Language code.

    Returns
    -------
    Tokenizer
        Ready-to-use BPE tokenizer.
    """
    tokenizer_file = config.tokenizer_file if hasattr(config, "tokenizer_file") else config["tokenizer_file"]
    vocab_size = config.vocab_size if hasattr(config, "vocab_size") else config.get("vocab_size", 20000)

    tokenizer_path = Path(tokenizer_file.format(lang))
    rank = int(os.environ.get("RANK", 0))

    if not tokenizer_path.exists():
        if rank == 0:
            logger.info("Rank 0: Building BPE tokenizer for %s (vocab_size=%d)...", lang, vocab_size)
            tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()

            trainer = BpeTrainer(
                special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
                min_frequency=2,
                vocab_size=vocab_size,
                show_progress=True,
            )

            tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
            tokenizer.save(str(tokenizer_path))
            logger.info("Tokenizer saved to %s", tokenizer_path)

    # Barrier: wait for Rank 0 to finish before other ranks load
    if dist.is_initialized():
        dist.barrier()

    return Tokenizer.from_file(str(tokenizer_path))


# ═══════════════════════════════════════════════════════════════════════
#  Dataset Construction
# ═══════════════════════════════════════════════════════════════════════


def get_ds(config) -> Tuple[BilingualDataset, BilingualDataset, Tokenizer, Tokenizer]:
    """Build training and validation datasets with tokenizers.

    DDP-safe: only Rank 0 downloads the dataset; others wait at barrier.

    Parameters
    ----------
    config : TransformerConfig or dict
        Training config with ``lang_src``, ``lang_tgt``, ``seq_len``.

    Returns
    -------
    tuple
        ``(train_ds, val_ds, tokenizer_src, tokenizer_tgt)``
    """
    lang_src = config.lang_src if hasattr(config, "lang_src") else config["lang_src"]
    lang_tgt = config.lang_tgt if hasattr(config, "lang_tgt") else config["lang_tgt"]
    seq_len = config.seq_len if hasattr(config, "seq_len") else config["seq_len"]

    rank = int(os.environ.get("RANK", 0))

    # DDP guard: only Rank 0 downloads first
    if rank == 0:
        load_dataset("opus_books", f"{lang_src}-{lang_tgt}", split="train")

    if dist.is_initialized():
        dist.barrier()

    ds_raw = load_dataset("opus_books", f"{lang_src}-{lang_tgt}", split="train")

    # Build or load tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, lang_src)
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, lang_tgt)

    # Train/validation split (90/10, deterministic seed)
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size

    generator = torch.Generator().manual_seed(42)
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size], generator=generator)

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, lang_src, lang_tgt, seq_len)
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, lang_src, lang_tgt, seq_len)

    logger.info(
        "Dataset ready: %d train / %d val | Src vocab: %d | Tgt vocab: %d",
        len(train_ds), len(val_ds),
        tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(),
    )

    return train_ds, val_ds, tokenizer_src, tokenizer_tgt
