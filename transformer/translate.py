"""
Interactive Translation Script
==============================

Standalone inference script for translating text using a trained
Transformer model.

Usage::

    # Single sentence
    python -m transformer.translate --checkpoint weights/tmodel_24.pt --text "Hello world"

    # Interactive mode
    python -m transformer.translate --checkpoint weights/tmodel_24.pt --interactive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

from transformer.config import TransformerConfig
from transformer.model import get_model
from transformer.validate import greedy_decode


def load_model_and_tokenizers(
    checkpoint_path: str,
    config: TransformerConfig,
    device: torch.device,
):
    """Load trained model and tokenizers from disk.

    Parameters
    ----------
    checkpoint_path : str
        Path to the model checkpoint (``.pt`` file).
    config : TransformerConfig
        Model configuration.
    device : torch.device
        Device to load model onto.

    Returns
    -------
    tuple
        ``(model, tokenizer_src, tokenizer_tgt)``
    """
    # Load tokenizers
    tokenizer_src_path = config.tokenizer_file.format(config.lang_src)
    tokenizer_tgt_path = config.tokenizer_file.format(config.lang_tgt)

    if not Path(tokenizer_src_path).exists():
        raise FileNotFoundError(f"Source tokenizer not found: {tokenizer_src_path}")
    if not Path(tokenizer_tgt_path).exists():
        raise FileNotFoundError(f"Target tokenizer not found: {tokenizer_tgt_path}")

    tokenizer_src = Tokenizer.from_file(tokenizer_src_path)
    tokenizer_tgt = Tokenizer.from_file(tokenizer_tgt_path)

    # Build model and load weights
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, tokenizer_src, tokenizer_tgt


def translate(
    model,
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    text: str,
    config: TransformerConfig,
    device: torch.device,
) -> str:
    """Translate a single sentence.

    Parameters
    ----------
    model : Transformer
        Trained model.
    tokenizer_src : Tokenizer
        Source language tokenizer.
    tokenizer_tgt : Tokenizer
        Target language tokenizer.
    text : str
        Input text in source language.
    config : TransformerConfig
        Config for seq_len.
    device : torch.device
        Compute device.

    Returns
    -------
    str
        Translated text in target language.
    """
    # Tokenize input
    sos_id = tokenizer_src.token_to_id("[SOS]")
    eos_id = tokenizer_src.token_to_id("[EOS]")
    pad_id = tokenizer_src.token_to_id("[PAD]")

    tokens = tokenizer_src.encode(text).ids
    tokens = tokens[: config.seq_len - 2]  # Leave room for [SOS] and [EOS]

    # Build encoder input: [SOS] + tokens + [EOS] + padding
    enc_tokens = [sos_id] + tokens + [eos_id]
    padding = [pad_id] * (config.seq_len - len(enc_tokens))
    encoder_input = torch.tensor(enc_tokens + padding, dtype=torch.int64).unsqueeze(0).to(device)

    # Encoder mask
    encoder_mask = (encoder_input != pad_id).unsqueeze(1).unsqueeze(1).int()

    # Greedy decode
    with torch.no_grad():
        output_tokens = greedy_decode(
            model, encoder_input, encoder_mask,
            tokenizer_src, tokenizer_tgt,
            config.seq_len, device,
        )

    # Decode token IDs to text
    output_text = tokenizer_tgt.decode(output_tokens.detach().cpu().numpy())
    return output_text


def main():
    parser = argparse.ArgumentParser(
        description="Translate text using a trained Transformer model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m transformer.translate --checkpoint weights/tmodel_24.pt --text "Hello world"
  python -m transformer.translate --checkpoint weights/tmodel_24.pt --interactive
        """,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--text", type=str, default=None, help="Text to translate")
    parser.add_argument("--interactive", action="store_true", help="Enter interactive translation mode")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config (optional)")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detects if not specified)")

    args = parser.parse_args()

    # Load config
    if args.config:
        config = TransformerConfig.from_yaml(args.config)
    else:
        config = TransformerConfig()

    # Auto-detect device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.checkpoint} on {device}...")

    model, tokenizer_src, tokenizer_tgt = load_model_and_tokenizers(
        args.checkpoint, config, device,
    )

    print(f"Model loaded! Translating {config.lang_src} → {config.lang_tgt}\n")

    if args.interactive:
        # Interactive mode
        print("=" * 50)
        print("Interactive Translation Mode")
        print(f"Type text in {config.lang_src.upper()} to translate to {config.lang_tgt.upper()}")
        print("Type 'quit' or 'exit' to stop")
        print("=" * 50)

        while True:
            try:
                text = input(f"\n[{config.lang_src.upper()}] > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if text.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if not text:
                continue

            result = translate(model, tokenizer_src, tokenizer_tgt, text, config, device)
            print(f"[{config.lang_tgt.upper()}] > {result}")

    elif args.text:
        result = translate(model, tokenizer_src, tokenizer_tgt, args.text, config, device)
        print(f"Source ({config.lang_src}):  {args.text}")
        print(f"Target ({config.lang_tgt}):  {result}")

    else:
        print("Error: Provide either --text or --interactive")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
