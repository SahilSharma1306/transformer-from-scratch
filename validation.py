import torch
import torch.nn as nn
from dataset import causal_mask

# Causal mask cache — never re-allocates the same tensor.
_mask_cache: dict = {}


def _get_causal_mask(size: int, device) -> torch.Tensor:
    if size not in _mask_cache:
        _mask_cache[size] = causal_mask(size)
    return _mask_cache[size].to(device)


def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    """
    Autoregressive greedy decoding with pre-allocated output buffer.

    Uses torch.inference_mode() for speed, cached causal masks to avoid
    tensor re-allocation, and in-place writes to a pre-allocated buffer
    (O(n) vs O(n²) with torch.cat per step).
    """
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    with torch.inference_mode():
        encoder_output = model.encode(source, source_mask)

        # Single pre-allocated buffer — write in-place.
        out_buf = torch.full((1, max_len), fill_value=sos_idx, dtype=torch.long, device=device)
        cur_len = 1

        while cur_len < max_len:
            tgt_mask = _get_causal_mask(cur_len, device).type_as(source_mask)
            dec_out = model.decode(encoder_output, source_mask, out_buf[:, :cur_len], tgt_mask)
            logits = model.project(dec_out[:, -1])  # project last token only
            next_tok = torch.argmax(logits, dim=-1).item()
            out_buf[0, cur_len] = next_tok
            cur_len += 1
            if next_tok == eos_idx:
                break

    return out_buf[0, :cur_len]


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len,
                   device, print_msg, global_step, writer, num_examples=2, loss_fn=None):
    """
    Two-pass validation: teacher-forced loss over all batches, then greedy
    decode for a few display examples.
    """
    model.eval()

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss(
            ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1,
        ).to(device)

    vocab_size = tokenizer_tgt.get_vocab_size()
    total_val_loss = 0.0
    count = 0
    display_cache = []

    with torch.inference_mode():
        for batch in validation_ds:
            count += 1
            enc_in = batch["encoder_input"].to(device, non_blocking=True)
            enc_msk = batch["encoder_mask"].to(device, non_blocking=True)
            dec_in = batch["decoder_input"].to(device, non_blocking=True)
            dec_msk = batch["decoder_mask"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            assert enc_in.size(0) == 1, "Validation batch size must be 1"

            enc_out = model.encode(enc_in, enc_msk)
            dec_out = model.decode(enc_out, enc_msk, dec_in, dec_msk)
            proj_out = model.project(dec_out)

            loss = loss_fn(proj_out.view(-1, vocab_size), label.view(-1))
            total_val_loss += loss.item()

            if count <= num_examples:
                display_cache.append({
                    "enc_in": enc_in, "enc_msk": enc_msk,
                    "src_text": batch["src_text"][0],
                    "tgt_text": batch["tgt_text"][0],
                })

    # Greedy decode for display examples.
    for db in display_cache:
        out = greedy_decode(model, db["enc_in"], db["enc_msk"],
                            tokenizer_tgt, max_len, device)
        predicted = tokenizer_tgt.decode(out.cpu().tolist())
        print_msg("-" * 80)
        print_msg(f"SOURCE:    {db['src_text']}")
        print_msg(f"TARGET:    {db['tgt_text']}")
        print_msg(f"PREDICTED: {predicted}")

    mean_val_loss = total_val_loss / max(count, 1)
    print_msg(f"Validation Loss: {mean_val_loss:.4f}")

    if writer:
        writer.add_scalar("val loss", mean_val_loss, global_step)
        writer.flush()

    return mean_val_loss
