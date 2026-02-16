"""
Training Loop — Distributed Data Parallel with Mixed Precision
===============================================================

Production-grade training pipeline with:
- **DDP** (DistributedDataParallel) for multi-GPU scaling
- **AMP** (Automatic Mixed Precision) with FP16 for ~2x throughput
- **Cosine LR schedule** with linear warmup
- **Gradient clipping** for training stability
- **Full checkpoint resume** (model + optimizer + scaler + step)
- **TensorBoard** logging for loss, gradient norms, and learning rate

Launch
------
Single GPU::

    python -m transformer.train

Multi-GPU (DDP)::

    torchrun --nproc_per_node=2 -m transformer.train

With config override::

    torchrun --nproc_per_node=2 -m transformer.train --config configs/default.yaml --num_epochs 50
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.tensorboard import SummaryWriter

from transformer.config import TransformerConfig
from transformer.dataset import get_ds
from transformer.model import get_model, model_summary
from transformer.validate import run_validation

# Suppress tokenizer parallelism warnings (conflicts with DataLoader workers)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable cuDNN auto-tuner for faster convolutions
torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)


def setup_logging(rank: int) -> None:
    """Configure logging (only rank 0 logs to console).

    Parameters
    ----------
    rank : int
        Global process rank.
    """
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def ddp_setup() -> None:
    """Initialize the DDP process group and set the local GPU."""
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Compute learning rate with linear warmup + cosine decay.

    Schedule::

        Step < warmup:   LR = max_lr * (step / warmup_steps)       [linear]
        Step >= warmup:  LR = min_lr + 0.5*(max_lr-min_lr)*(1+cos) [cosine]

    Parameters
    ----------
    step : int
        Current training step.
    warmup_steps : int
        Number of warmup steps.
    max_steps : int
        Total training steps.
    max_lr : float
        Peak learning rate.
    min_lr : float
        Minimum learning rate floor.

    Returns
    -------
    float
        Learning rate for this step.
    """
    if step < warmup_steps:
        return max_lr * ((step + 1) / warmup_steps)

    if step > max_steps:
        return min_lr

    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    decay_ratio = max(0.0, min(1.0, decay_ratio))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (max_lr - min_lr)


def train_model(config: TransformerConfig) -> None:
    """Main training function.

    Parameters
    ----------
    config : TransformerConfig
        Complete training configuration.
    """
    ddp_setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    device = torch.device(f"cuda:{local_rank}")

    setup_logging(global_rank)

    if global_rank == 0:
        logger.info("Master process started. Checkpoints → %s", config.model_folder)
        Path(config.model_folder).mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────
    train_ds, val_ds, tokenizer_src, tokenizer_tgt = get_ds(config)

    train_sampler = DistributedSampler(train_ds)
    train_dataloader = DataLoader(
        dataset=train_ds,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=config.num_workers,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    # ── Model ────────────────────────────────────────────────────────
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    if global_rank == 0:
        logger.info("\n%s", model_summary(model))

    model = DDP(model, device_ids=[local_rank])

    # ── Optimizer ────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        eps=1e-9,
        weight_decay=config.weight_decay,
    )

    # ── Mixed Precision ──────────────────────────────────────────────
    scaler = torch.amp.GradScaler("cuda")

    # ── LR Schedule ──────────────────────────────────────────────────
    max_steps = int(len(train_dataloader) * config.num_epochs * 0.9)
    warmup_steps = int(max_steps * config.warmup_pct)
    max_lr = config.lr
    min_lr = max_lr * config.min_lr_ratio

    if global_rank == 0:
        logger.info("Training: %d steps total, %d warmup, LR %.1e → %.1e",
                     max_steps, warmup_steps, max_lr, min_lr)

    # ── TensorBoard ──────────────────────────────────────────────────
    writer = SummaryWriter(config.experiment_name) if global_rank == 0 else None

    # ── Resume from Checkpoint ───────────────────────────────────────
    initial_epoch = 0
    global_step = 0

    history = {"train_loss": [], "val_loss": [], "grad_norm": [], "epochs": []}

    if config.preload:
        model_filename = config.get_weights_file_path(config.preload)
        if global_rank == 0:
            logger.info("Resuming from checkpoint: %s", model_filename)

        state = torch.load(model_filename, map_location=device)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
        model.module.load_state_dict(state["model_state_dict"])

        if "scaler_state_dict" in state:
            scaler.load_state_dict(state["scaler_state_dict"])

    # ── Loss Function ────────────────────────────────────────────────
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_tgt.token_to_id("[PAD]"),
        label_smoothing=config.label_smoothing,
    ).to(device)

    # ── Training Loop ────────────────────────────────────────────────
    for epoch in range(initial_epoch, config.num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()

        if global_rank == 0:
            logger.info("━" * 60)
            logger.info("Epoch %d / %d", epoch, config.num_epochs - 1)
            logger.info("━" * 60)

        epoch_train_losses = []
        epoch_grad_norms = []
        start_time = time.time()

        for batch in train_dataloader:
            # Dynamic learning rate
            lr = get_lr(global_step, warmup_steps, max_steps, max_lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            # Move data to GPU
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)

            # Forward pass (mixed precision)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                encoder_output = model.module.encode(encoder_input, encoder_mask)
                decoder_output = model.module.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                proj_output = model.module.project(decoder_output)
                loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

            # Backward pass (mixed precision)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            current_loss = loss.item()
            current_norm = norm.item()
            epoch_train_losses.append(current_loss)
            epoch_grad_norms.append(current_norm)

            # Periodic logging
            if global_step % 50 == 0 and global_rank == 0:
                elapsed = time.time() - start_time
                speed = 50 / elapsed if elapsed > 0 else 0

                logger.info(
                    "Step %5d │ Loss: %.4f │ LR: %.2e │ Norm: %.4f │ Speed: %.1f steps/s",
                    global_step, current_loss, lr, current_norm, speed,
                )

                if writer:
                    writer.add_scalar("train loss", current_loss, global_step)
                    writer.add_scalar("grad norm", current_norm, global_step)
                    writer.add_scalar("learning rate", lr, global_step)

                start_time = time.time()

            global_step += 1

        # ── End of Epoch: Validation & Checkpointing ─────────────────
        if global_rank == 0:
            logger.info("Running validation...")
            val_loss = run_validation(
                model.module, val_dataloader, tokenizer_src, tokenizer_tgt,
                config.seq_len, device, lambda msg: logger.info(msg),
                global_step, writer,
            )

            avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms)

            history["train_loss"].append(avg_train_loss)
            history["grad_norm"].append(avg_grad_norm)
            history["val_loss"].append(val_loss)
            history["epochs"].append(epoch)

            # Save checkpoint
            model_filename = config.get_weights_file_path(f"{epoch:02d}")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "global_step": global_step,
            }, model_filename)
            logger.info("Checkpoint saved: %s", model_filename)

            # Save training history as JSON
            history_file = os.path.join(config.model_folder, "training_history.json")
            with open(history_file, "w") as f:
                json.dump(history, f, indent=2)

    # ── Cleanup ──────────────────────────────────────────────────────
    if writer:
        writer.close()
    destroy_process_group()

    if global_rank == 0:
        logger.info("Training complete! Final val loss: %.4f", history["val_loss"][-1])


# ═══════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = TransformerConfig.from_cli()
    train_model(config)
