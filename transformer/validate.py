"""
Validation & Inference Utilities
=================================

Provides greedy decoding for autoregressive inference and a full
validation loop with loss computation and sample translations.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch
import torch.nn as nn
from tokenizers import Tokenizer

from transformer.dataset import causal_mask

logger = logging.getLogger(__name__)


def greedy_decode(
    model: nn.Module,
    source: torch.Tensor,
    source_mask: torch.Tensor,
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate a translation using greedy (argmax) decoding.

    Starting from [SOS], iteratively picks the most probable next
    token until [EOS] is generated or ``max_len`` is reached.

    Parameters
    ----------
    model : nn.Module
        Trained Transformer model.
    source : torch.Tensor
        Encoded source input ``(1, T_src)``.
    source_mask : torch.Tensor
        Source padding mask.
    tokenizer_src : Tokenizer
        Source tokenizer (unused, kept for API consistency).
    tokenizer_tgt : Tokenizer
        Target tokenizer (for [SOS]/[EOS] token IDs).
    max_len : int
        Maximum output sequence length.
    device : torch.device
        Device to run inference on.

    Returns
    -------
    torch.Tensor
        Generated token IDs ``(T_out,)``.
    """
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # Encode source once
    encoder_output = model.encode(source, source_mask)

    # Start with [SOS]
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build causal mask for current decoder length
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Decode
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Project last position to vocabulary
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)

        # Append predicted token
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)],
            dim=1,
        )

        # Stop at [EOS]
        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(
    model: nn.Module,
    validation_ds,
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    max_len: int,
    device: torch.device,
    print_msg: Callable[[str], None],
    global_step: int,
    writer=None,
    num_examples: int = 2,
) -> float:
    """Run validation: compute loss and show sample translations.

    Parameters
    ----------
    model : nn.Module
        Trained Transformer model (should be in eval mode).
    validation_ds : DataLoader
        Validation DataLoader (batch_size=1).
    tokenizer_src : Tokenizer
        Source tokenizer.
    tokenizer_tgt : Tokenizer
        Target tokenizer.
    max_len : int
        Maximum decoding length.
    device : torch.device
        Compute device.
    print_msg : callable
        Function to output messages (e.g. ``logger.info``).
    global_step : int
        Current training step (for TensorBoard logging).
    writer : SummaryWriter, optional
        TensorBoard writer.
    num_examples : int
        Number of sample translations to display.

    Returns
    -------
    float
        Mean validation loss across all batches.
    """
    model.eval()
    count = 0

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_tgt.token_to_id("[PAD]"),
        label_smoothing=0.1,
    ).to(device)

    total_val_loss = 0.0

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            # Full forward pass
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            total_val_loss += loss.item()

            # Show sample translations
            if count <= num_examples:
                model_out = greedy_decode(
                    model, encoder_input, encoder_mask,
                    tokenizer_src, tokenizer_tgt, max_len, device,
                )
                source_text = batch["src_text"][0]
                target_text = batch["tgt_text"][0]
                model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

                print_msg("-" * 80)
                print_msg(f"SOURCE:    {source_text}")
                print_msg(f"TARGET:    {target_text}")
                print_msg(f"PREDICTED: {model_out_text}")

    mean_val_loss = total_val_loss / count
    print_msg(f"Validation Loss: {mean_val_loss:.4f}")

    if writer:
        writer.add_scalar("val loss", mean_val_loss, global_step)
        writer.flush()

    return mean_val_loss
