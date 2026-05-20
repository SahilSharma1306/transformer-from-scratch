import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import contextlib
import warnings
from pathlib import Path
import numpy as np
import time
import math

from config import get_config, get_weights_file_path
from dataset import get_ds
from model import get_model
from validation import run_validation

torch.backends.cudnn.benchmark = True


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def train_model(config):
    ddp_setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    device = torch.device(f"cuda:{local_rank}")

    if global_rank == 0:
        print(f"Master process started. Saving to {config['model_folder']}")
        Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, tokenizer_src, tokenizer_tgt = get_ds(config)

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_dataloader = DataLoader(
        dataset=train_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True,
                                pin_memory=True, num_workers=2)

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    model = DDP(model, device_ids=[local_rank])

    # AdamW fused=True — single CUDA kernel for all param updates.
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["lr"],
            eps=1e-8, weight_decay=0.1, fused=True,
        )
    except TypeError:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["lr"], eps=1e-8, weight_decay=0.1,
        )

    amp_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda")
    if global_rank == 0:
        print(f"AMP dtype: {amp_dtype} | GradScaler: on")

    # ── LR schedule ──────────────────────────────────────────────────────────
    GRAD_ACCUM = config.get("grad_accum_steps", 1)
    steps_per_epoch = math.ceil(len(train_dataloader) / GRAD_ACCUM)
    MAX_STEPS = int(steps_per_epoch * config["num_epochs"] * 0.9)
    WARMUP_STEPS = max(1, int(MAX_STEPS * 0.05))
    MAX_LR, MIN_LR = 6e-4, 6e-5

    if global_rank == 0:
        print(f"Optimiser steps: {MAX_STEPS} | Warmup: {WARMUP_STEPS} | Accum: {GRAD_ACCUM}")

    def get_lr(step):
        if step < WARMUP_STEPS:
            return MAX_LR * (step + 1) / WARMUP_STEPS
        if step >= MAX_STEPS:
            return MIN_LR
        ratio = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
        return MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1.0 + math.cos(math.pi * ratio))

    # Loss fn built ONCE, reused in validation
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1,
    ).to(device)

    tgt_vocab_size = tokenizer_tgt.get_vocab_size()
    writer = SummaryWriter(config["experiment_name"]) if global_rank == 0 else None
    initial_epoch = 0
    global_step = 0
    history = {"train_loss": [], "val_loss": [], "grad_norm": [], "epochs": []}

    if config["preload"]:
        fn = get_weights_file_path(config, config["preload"])
        state = torch.load(fn, map_location=device)
        initial_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        optimizer.load_state_dict(state["optimizer_state_dict"])
        model.module.load_state_dict(state["model_state_dict"])
        if scaler and "scaler_state_dict" in state:
            scaler.load_state_dict(state["scaler_state_dict"])
        if global_rank == 0:
            print(f"Resumed from {fn}")

    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    # ── Main loop ─────────────────────────────────────────────────────────────
    for epoch in range(initial_epoch, config["num_epochs"]):
        train_sampler.set_epoch(epoch)
        model.train()
        if global_rank == 0:
            print(f"\n{'=' * 60}\nEpoch {epoch}\n{'=' * 60}")

        epoch_losses = []
        epoch_norms = []
        start_time = time.time()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_dataloader):
            enc_in = batch["encoder_input"].to(device, non_blocking=True)
            dec_in = batch["decoder_input"].to(device, non_blocking=True)
            enc_msk = batch["encoder_mask"].to(device, non_blocking=True)
            dec_msk = batch["decoder_mask"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            is_last_accum = ((batch_idx + 1) % GRAD_ACCUM == 0) or (batch_idx + 1 == len(train_dataloader))

            # no_sync() suppresses AllReduce on micro-batches
            sync_ctx = contextlib.nullcontext() if is_last_accum else model.no_sync()

            with sync_ctx:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    proj_output = model(enc_in, enc_msk, dec_in, dec_msk)
                    loss = loss_fn(
                        proj_output.view(-1, tgt_vocab_size), label.view(-1)
                    ) / GRAD_ACCUM

                scaler.scale(loss).backward()

            if is_last_accum:
                lr = get_lr(global_step)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                raw_loss = loss.item() * GRAD_ACCUM
                raw_norm = norm.item()
                epoch_losses.append(raw_loss)
                epoch_norms.append(raw_norm)

                if global_step % 50 == 0 and global_rank == 0:
                    elapsed = time.time() - start_time
                    speed = 50 / elapsed if elapsed > 0 else 0
                    print(f"  Step {global_step:6d} | Loss: {raw_loss:.4f} | "
                          f"LR: {lr:.2e} | Norm: {raw_norm:.3f} | {speed:.1f} steps/s")
                    if writer:
                        writer.add_scalar("train loss", raw_loss, global_step)
                        writer.add_scalar("grad norm", raw_norm, global_step)
                        writer.add_scalar("lr", lr, global_step)
                    start_time = time.time()

                global_step += 1

        # ── Validation + checkpoint ───────────────────────────────────────
        if global_rank == 0:
            val_loss = run_validation(
                model.module, val_dataloader, tokenizer_src, tokenizer_tgt,
                config["seq_len"], device, print, global_step, writer,
                loss_fn=loss_fn,
            )
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_norm = sum(epoch_norms) / len(epoch_norms)
            history["train_loss"].append(avg_loss)
            history["val_loss"].append(val_loss)
            history["grad_norm"].append(avg_norm)
            history["epochs"].append(epoch)
            print(f"Epoch {epoch} | Train: {avg_loss:.4f} | Val: {val_loss:.4f}")

            ckpt = {
                "epoch": epoch, "global_step": global_step,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            }
            torch.save(ckpt, get_weights_file_path(config, f"{epoch:02d}"))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(ckpt, get_weights_file_path(config, 'best'))
                print(f"  New best saved (val={val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{patience}")

            hist_path = os.path.join(config["model_folder"], f"history_epoch_{epoch:02d}.npy")
            np.save(hist_path, {
                "train_loss": np.array(history["train_loss"]),
                "val_loss": np.array(history["val_loss"]),
                "grad_norm": np.array(history["grad_norm"]),
            })
            print(f"Saved history -> {hist_path}")

        # Broadcast early stopping decision to all ranks
        should_stop = torch.tensor(patience_counter, dtype=torch.int32).to(device)
        if dist.is_initialized():
            dist.broadcast(should_stop, src=0)
        if should_stop.item() >= patience:
            if global_rank == 0:
                print(f"Early stopping triggered at epoch {epoch}")
            break

    destroy_process_group()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
