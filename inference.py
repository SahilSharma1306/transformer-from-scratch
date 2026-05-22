"""
Standalone English → French translation script.

Usage
-----
    python inference.py --text "The weather is beautiful today."
    python inference.py --text "I love learning new things." --checkpoint weights/tmodel_11.pt

The script loads the trained Transformer checkpoint and BPE tokenizers,
then performs autoregressive greedy decoding to produce a French translation.
"""

import argparse
import torch
from pathlib import Path
from tokenizers import Tokenizer

from config import get_config
from model import get_model
from dataset import causal_mask


# ─── Causal-mask cache ────────────────────────────────────────────────
_mask_cache: dict = {}


def _get_causal_mask(size: int, device: torch.device) -> torch.Tensor:
    if size not in _mask_cache:
        _mask_cache[size] = causal_mask(size)
    return _mask_cache[size].to(device)


# ─── Greedy decode (mirror of validation.py) ──────────────────────────
@torch.inference_mode()
def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    """Autoregressive greedy decoding with pre-allocated output buffer."""
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encode(source, source_mask)

    out_buf = torch.full((1, max_len), fill_value=sos_idx,
                         dtype=torch.long, device=device)
    cur_len = 1

    while cur_len < max_len:
        tgt_mask = _get_causal_mask(cur_len, device).type_as(source_mask)
        dec_out = model.decode(encoder_output, source_mask,
                               out_buf[:, :cur_len], tgt_mask)
        logits = model.project(dec_out[:, -1])
        next_tok = torch.argmax(logits, dim=-1).item()
        out_buf[0, cur_len] = next_tok
        cur_len += 1
        if next_tok == eos_idx:
            break

    return out_buf[0, :cur_len]


# ─── Translate ────────────────────────────────────────────────────────
def translate(text: str, model, tokenizer_src, tokenizer_tgt,
              seq_len: int, device: torch.device) -> str:
    """Encode a source sentence and decode the French translation."""
    model.eval()

    sos = tokenizer_src.token_to_id("[SOS]")
    eos = tokenizer_src.token_to_id("[EOS]")
    pad = tokenizer_src.token_to_id("[PAD]")

    # Tokenize and build encoder input
    enc_ids = tokenizer_src.encode(text).ids[:seq_len - 2]
    enc_input = (
        [sos] + enc_ids + [eos] + [pad] * (seq_len - len(enc_ids) - 2)
    )

    source = torch.tensor([enc_input], dtype=torch.long, device=device)
    source_mask = (source != pad).unsqueeze(1).unsqueeze(1).int()

    # Decode
    out_tokens = greedy_decode(model, source, source_mask,
                               tokenizer_tgt, seq_len, device)
    return tokenizer_tgt.decode(out_tokens.cpu().tolist())


# ─── CLI ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Translate English text to French using the trained Transformer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --text "The weather is beautiful today."
  python inference.py --text "I love learning." --checkpoint weights/tmodel_11.pt
        """,
    )
    parser.add_argument(
        "--text", type=str, required=True,
        help="English sentence to translate.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (.pt). Auto-detects the latest if omitted.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to run inference on (default: auto-detect cuda/cpu).",
    )
    args = parser.parse_args()

    config = get_config()
    seq_len = config["seq_len"]

    # ── Device ────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Tokenizers ────────────────────────────────────────────────────
    tok_src_path = config["tokenizer_file"].format(config["lang_src"])
    tok_tgt_path = config["tokenizer_file"].format(config["lang_tgt"])

    if not Path(tok_src_path).exists() or not Path(tok_tgt_path).exists():
        print(f"ERROR: Tokenizer files not found.")
        print(f"  Expected: {tok_src_path} and {tok_tgt_path}")
        print(f"  Run training first, or download the pre-built tokenizers.")
        return

    tokenizer_src = Tokenizer.from_file(tok_src_path)
    tokenizer_tgt = Tokenizer.from_file(tok_tgt_path)

    # ── Model ─────────────────────────────────────────────────────────
    model = get_model(config, tokenizer_src.get_vocab_size(),
                      tokenizer_tgt.get_vocab_size()).to(device)

    # ── Checkpoint ────────────────────────────────────────────────────
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        # Auto-detect the latest checkpoint
        weights_dir = Path(config["model_folder"])
        if weights_dir.exists():
            ckpts = sorted(weights_dir.glob(f"{config['model_basename']}*.pt"))
            if ckpts:
                ckpt_path = str(ckpts[-1])

    if ckpt_path is None or not Path(ckpt_path).exists():
        print(f"ERROR: No checkpoint found.")
        print(f"  Looked in: {config['model_folder']}/")
        print(f"  Train the model first or specify --checkpoint <path>")
        return

    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Handle DDP-wrapped checkpoints (keys prefixed with "module.")
    state_dict = state.get("model_state_dict", state)
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned)

    # ── Translate ─────────────────────────────────────────────────────
    print(f"Device: {device}")
    print(f"{'─' * 60}")
    print(f"  EN: {args.text}")

    translation = translate(args.text, model, tokenizer_src,
                            tokenizer_tgt, seq_len, device)
    print(f"  FR: {translation}")
    print(f"{'─' * 60}")


if __name__ == "__main__":
    main()
