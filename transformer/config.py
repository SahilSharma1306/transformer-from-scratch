"""
Configuration System
====================

Defines the ``TransformerConfig`` dataclass that centralizes every
hyperparameter, path, and training knob.  Supports three override layers:

    defaults → YAML file → CLI arguments

Usage::

    # From code
    cfg = TransformerConfig()

    # From YAML
    cfg = TransformerConfig.from_yaml("configs/default.yaml")

    # From CLI (auto-generates argparse flags for every field)
    cfg = TransformerConfig.from_cli()
"""

from __future__ import annotations

import argparse
import yaml
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Optional


@dataclass
class TransformerConfig:
    """Complete configuration for the Transformer model and training pipeline.

    Attributes
    ----------
    batch_size : int
        Number of sentence pairs per GPU per training step.
    num_epochs : int
        Total training epochs.
    lr : float
        Peak learning rate (reached after warmup).
    seq_len : int
        Maximum sequence length for both source and target.
    d_model : int
        Dimensionality of the model's hidden representations.
    num_heads : int
        Number of attention heads.  ``d_model`` must be divisible by this.
    num_layers : int
        Number of encoder (and decoder) blocks stacked.
    dropout : float
        Dropout probability applied after attention and FFN layers.
    lang_src : str
        Source language ISO code (e.g. ``"en"``).
    lang_tgt : str
        Target language ISO code (e.g. ``"it"``).
    model_folder : str
        Directory to save/load model checkpoints.
    model_basename : str
        Prefix for checkpoint filenames (e.g. ``"tmodel_"``).
    preload : str or None
        Epoch string to resume training from (e.g. ``"04"``).  None = train
        from scratch.
    tokenizer_file : str
        Template for tokenizer save paths.  ``{0}`` is replaced with the
        language code.
    experiment_name : str
        TensorBoard log directory.
    warmup_pct : float
        Fraction of total training steps used for linear LR warmup.
    min_lr_ratio : float
        Minimum LR as a fraction of peak LR (for cosine decay floor).
    weight_decay : float
        AdamW weight decay coefficient.
    max_grad_norm : float
        Maximum gradient norm for clipping.
    label_smoothing : float
        Label smoothing factor for cross-entropy loss.
    vocab_size : int
        BPE tokenizer vocabulary size.
    rope_theta : float
        Base frequency for Rotary Positional Embeddings.
    num_workers : int
        DataLoader worker processes.
    """

    # ── Model Architecture ───────────────────────────────────────────
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.3
    seq_len: int = 320
    rope_theta: float = 10000.0

    # ── Training ─────────────────────────────────────────────────────
    batch_size: int = 32
    num_epochs: int = 25
    lr: float = 6e-4
    warmup_pct: float = 0.05
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1
    num_workers: int = 8

    # ── Data ─────────────────────────────────────────────────────────
    lang_src: str = "en"
    lang_tgt: str = "it"
    vocab_size: int = 20000

    # ── Paths ────────────────────────────────────────────────────────
    model_folder: str = "weights"
    model_basename: str = "tmodel_"
    preload: Optional[str] = None
    tokenizer_file: str = "tokenizer_{0}.json"
    experiment_name: str = "runs/tmodel"

    # ── Helpers ──────────────────────────────────────────────────────

    def get_weights_file_path(self, epoch: str | int) -> str:
        """Return the full path for a checkpoint at a given epoch.

        Parameters
        ----------
        epoch : str or int
            Epoch identifier (e.g. ``11`` or ``"11"``).

        Returns
        -------
        str
            Absolute path like ``weights/tmodel_11.pt``.
        """
        model_filename = f"{self.model_basename}{epoch}.pt"
        return str(Path(self.model_folder) / model_filename)

    def get_latest_weights_file_path(self) -> Optional[str]:
        """Find the most recent checkpoint in ``model_folder``.

        Returns
        -------
        str or None
            Path to the latest checkpoint, or None if no checkpoints exist.
        """
        folder = Path(self.model_folder)
        if not folder.exists():
            return None
        weights = sorted(folder.glob(f"{self.model_basename}*.pt"))
        return str(weights[-1]) if weights else None

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Convert config to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TransformerConfig":
        """Load config from a YAML file, falling back to defaults for
        any missing keys.

        Parameters
        ----------
        path : str or Path
            Path to the YAML configuration file.
        """
        with open(path, "r") as f:
            overrides = yaml.safe_load(f) or {}
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in overrides.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_cli(cls) -> "TransformerConfig":
        """Build config from command-line arguments.

        Every dataclass field becomes a CLI flag.  If ``--config`` is
        provided, YAML values are loaded first, then CLI flags override.

        Returns
        -------
        TransformerConfig
            Merged configuration.
        """
        parser = argparse.ArgumentParser(
            description="Transformer Training Configuration",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "--config", type=str, default=None,
            help="Path to YAML config file (values override defaults, CLI overrides YAML)",
        )

        # Auto-generate a flag for every dataclass field
        for f in fields(cls):
            flag = f"--{f.name}"
            if f.type == "bool" or f.type is bool:
                parser.add_argument(flag, type=_str_to_bool, default=None)
            elif f.type in ("Optional[str]", "str | None"):
                parser.add_argument(flag, type=str, default=None)
            elif f.type in ("int",) or f.type is int:
                parser.add_argument(flag, type=int, default=None)
            elif f.type in ("float",) or f.type is float:
                parser.add_argument(flag, type=float, default=None)
            else:
                parser.add_argument(flag, type=str, default=None)

        args = parser.parse_args()

        # Layer 1: defaults
        config_dict = {}

        # Layer 2: YAML overrides
        if args.config:
            with open(args.config, "r") as f:
                yaml_cfg = yaml.safe_load(f) or {}
            valid_fields = {f.name for f in fields(cls)}
            config_dict.update({k: v for k, v in yaml_cfg.items() if k in valid_fields})

        # Layer 3: CLI overrides (only non-None values)
        for f in fields(cls):
            cli_val = getattr(args, f.name, None)
            if cli_val is not None:
                config_dict[f.name] = cli_val

        return cls(**config_dict)


def _str_to_bool(v: str) -> bool:
    """Parse boolean CLI arguments flexibly."""
    if v.lower() in ("true", "1", "yes"):
        return True
    elif v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")
