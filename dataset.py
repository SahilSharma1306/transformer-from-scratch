import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
import torch.distributed as dist
import numpy as np


def causal_mask(size: int) -> torch.Tensor:
    """Upper-triangular mask: 1 = attend, 0 = block future positions."""
    return torch.triu(torch.ones(1, size, size, dtype=torch.bool), diagonal=1).logical_not()


class BilingualDataset(Dataset):
    """
    Reads pre-tokenized numpy arrays from disk. __getitem__ only does tensor
    construction — no tokeniser calls, no Python string ops.
    """

    def __init__(
        self,
        indices,
        src_ids_cache: np.ndarray,
        tgt_ids_cache: np.ndarray,
        src_texts: list,
        tgt_texts: list,
        tokenizer_src,
        tokenizer_tgt,
        seq_len: int,
    ):
        self.indices = indices
        self.src_ids = src_ids_cache
        self.tgt_ids = tgt_ids_cache
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.seq_len = seq_len

        self.pad_src = tokenizer_src.token_to_id('[PAD]')
        self.pad_tgt = tokenizer_tgt.token_to_id('[PAD]')
        self.sos_src = tokenizer_src.token_to_id('[SOS]')
        self.eos_src = tokenizer_src.token_to_id('[EOS]')
        self.sos_tgt = tokenizer_tgt.token_to_id('[SOS]')
        self.eos_tgt = tokenizer_tgt.token_to_id('[EOS]')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        # Grab pre-tokenized int arrays and truncate if needed
        enc_ids = self.src_ids[i][:self.seq_len - 2]
        dec_ids = self.tgt_ids[i][:self.seq_len - 1]

        enc_pad = self.seq_len - len(enc_ids) - 2
        dec_pad = self.seq_len - len(dec_ids) - 1

        # Encoder input: [SOS] + tokens + [EOS] + <PAD…>
        encoder_input = torch.cat([
            torch.tensor([self.sos_src], dtype=torch.int64),
            torch.tensor(enc_ids, dtype=torch.int64),
            torch.tensor([self.eos_src], dtype=torch.int64),
            torch.full((enc_pad,), self.pad_src, dtype=torch.int64),
        ])

        # Decoder input: [SOS] + tokens + <PAD…>
        decoder_input = torch.cat([
            torch.tensor([self.sos_tgt], dtype=torch.int64),
            torch.tensor(dec_ids, dtype=torch.int64),
            torch.full((dec_pad,), self.pad_tgt, dtype=torch.int64),
        ])

        # Label: tokens + [EOS] + <PAD…>
        label = torch.cat([
            torch.tensor(dec_ids, dtype=torch.int64),
            torch.tensor([self.eos_tgt], dtype=torch.int64),
            torch.full((dec_pad,), self.pad_tgt, dtype=torch.int64),
        ])

        enc_mask = (encoder_input != self.pad_src).unsqueeze(0).unsqueeze(0).int()
        dec_mask = (decoder_input != self.pad_tgt).unsqueeze(0).int() & causal_mask(self.seq_len)

        return {
            "encoder_input": encoder_input,   # (seq_len,)
            "decoder_input": decoder_input,   # (seq_len,)
            "encoder_mask":  enc_mask,        # (1,1,seq_len)
            "decoder_mask":  dec_mask,        # (1,seq_len,seq_len)
            "label":         label,           # (seq_len,)
            "src_text":      self.src_texts[i],
            "tgt_text":      self.tgt_texts[i],
        }


def _all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang) -> Tokenizer:
    path = Path(config['tokenizer_file'].format(lang))
    rank = int(os.environ.get("RANK", 0))

    if not path.exists():
        if rank == 0:
            print(f"[Rank 0] Building BPE tokenizer for '{lang}'…")
            tok = Tokenizer(BPE(unk_token='[UNK]'))
            tok.pre_tokenizer = Whitespace()
            trainer = BpeTrainer(
                special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
                min_frequency=2,
                vocab_size=20_000,
                show_progress=True,
            )
            tok.train_from_iterator(_all_sentences(ds, lang), trainer=trainer)
            tok.save(str(path))
            print(f"[Rank 0] Tokenizer saved → {path}")

    if dist.is_initialized():
        dist.barrier()

    return Tokenizer.from_file(str(path))


def _build_or_load_cache(config, ds_raw, tokenizer_src, tokenizer_tgt):
    cache_dir = Path(config['tokenized_cache_dir'])
    src_path = cache_dir / "src_ids.npy"
    tgt_path = cache_dir / "tgt_ids.npy"
    rank = int(os.environ.get("RANK", 0))

    if not src_path.exists():
        if rank == 0:
            print("[Rank 0] Pre-tokenising dataset (once)…")
            cache_dir.mkdir(parents=True, exist_ok=True)

            src_lang = config['lang_src']
            tgt_lang = config['lang_tgt']

            src_sentences = [item['translation'][src_lang] for item in ds_raw]
            tgt_sentences = [item['translation'][tgt_lang] for item in ds_raw]

            # Batch encode — returns list[Encoding] in Rust threads
            src_encodings = tokenizer_src.encode_batch(src_sentences)
            tgt_encodings = tokenizer_tgt.encode_batch(tgt_sentences)

            src_ids = np.empty(len(src_encodings), dtype=object)
            tgt_ids = np.empty(len(tgt_encodings), dtype=object)
            for i, (se, te) in enumerate(zip(src_encodings, tgt_encodings)):
                src_ids[i] = np.array(se.ids, dtype=np.int32)
                tgt_ids[i] = np.array(te.ids, dtype=np.int32)

            np.save(str(src_path), src_ids, allow_pickle=True)
            np.save(str(tgt_path), tgt_ids, allow_pickle=True)
            print(f"[Rank 0] Cache saved → {cache_dir}")

    if dist.is_initialized():
        dist.barrier()

    src_ids = np.load(str(src_path), allow_pickle=True)
    tgt_ids = np.load(str(tgt_path), allow_pickle=True)
    return src_ids, tgt_ids


def get_ds(config):
    rank = int(os.environ.get("RANK", 0))
    lang_pair = f"{config['lang_src']}-{config['lang_tgt']}"

    # Only Rank 0 downloads; barrier ensures others wait.
    if rank == 0:
        load_dataset('opus_books', lang_pair, split='train')
        load_dataset('opus100', lang_pair, split='train')
    if dist.is_initialized():
        dist.barrier()

    opus_books = load_dataset('opus_books', lang_pair, split='train')
    opus_100 = load_dataset('opus100', lang_pair, split='train')
    ds_raw = concatenate_datasets([opus_books, opus_100])

    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    src_ids_cache, tgt_ids_cache = _build_or_load_cache(
        config, ds_raw, tokenizer_src, tokenizer_tgt
    )

    # Pre-extract text strings (cheap, used only for display in validation)
    src_texts = [item['translation'][config['lang_src']] for item in ds_raw]
    tgt_texts = [item['translation'][config['lang_tgt']] for item in ds_raw]

    n = len(ds_raw)
    val_size = 10000
    train_size = n - val_size

    # Reproducible split using a fixed seed
    gen = torch.Generator().manual_seed(42)
    all_indices = torch.randperm(n, generator=gen).tolist()
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:]

    train_ds = BilingualDataset(
        train_indices, src_ids_cache, tgt_ids_cache,
        src_texts, tgt_texts, tokenizer_src, tokenizer_tgt, config['seq_len']
    )
    val_ds = BilingualDataset(
        val_indices, src_ids_cache, tgt_ids_cache,
        src_texts, tgt_texts, tokenizer_src, tokenizer_tgt, config['seq_len']
    )

    return train_ds, val_ds, tokenizer_src, tokenizer_tgt
