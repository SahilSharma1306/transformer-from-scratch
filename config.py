from pathlib import Path


def get_config():
    return {
        "lang_src": "en",
        "lang_tgt": "fr",
        "seq_len": 128,
        "d_model": 512,
        "dropout": 0.3,
        "batch_size": 64,
        "grad_accum_steps": 4,
        "num_epochs": 15,
        "lr": 3e-4,
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "tokenized_cache_dir": "tokenized_cache",
        "experiment_name": "runs/tmodel",
    }


def get_weights_file_path(config, epoch: str) -> str:
    return str(
        Path(config["model_folder"]) / f"{config['model_basename']}{epoch}.pt"
    )
