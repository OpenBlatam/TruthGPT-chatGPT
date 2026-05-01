from typing import Any, Dict, List

from optimization_core.build import build_components
from optimization_core.factories.callbacks import CALLBACKS
from optimization_core.trainers.trainer import GenericTrainer
from optimization_core.trainers.config import TrainerConfig


def build_trainer(
    raw_cfg: Dict[str, Any],
    train_texts: List[str],
    val_texts: List[str],
    max_seq_len: int
) -> GenericTrainer:
    comps = build_components(raw_cfg)
    memory_manager = comps.get("memory_manager")
    if memory_manager is not None:
        try:
            memory_manager.configure_matmul_precision()
        except Exception:
            pass
    # Build callbacks from training.callbacks
    training_cfg = raw_cfg.get("training", {})
    callback_names = training_cfg.get("callbacks", []) or []
    callbacks = []
    log_cfg = raw_cfg.get("logging", {})
    for name in callback_names:
        try:
            if name == "wandb":
                callbacks.append(CALLBACKS.build("wandb", project=log_cfg.get("project"), run_name=log_cfg.get("run_name")))
            elif name == "tensorboard":
                callbacks.append(CALLBACKS.build("tensorboard", log_dir=log_cfg.get("dir")))
            else:
                callbacks.append(CALLBACKS.build(name))
        except Exception:
            continue

    # Optionally construct datasets from YAML if external lists not provided
    if not train_texts or not val_texts:
        data_cfg = raw_cfg.get("data", {})
        source = str(data_cfg.get("source", "hf"))
        text_field = str(data_cfg.get("text_field", "text"))
        streaming = bool(data_cfg.get("streaming", False))
        if source == "hf":
            train_texts, val_texts = comps["datasets"].get("hf")(
                data_cfg.get("dataset", "wikitext"), data_cfg.get("subset", None), text_field, streaming, None
            )
        elif source == "jsonl":
            train_texts, val_texts = comps["datasets"].get("jsonl")(data_cfg.get("path", "data.jsonl"), text_field, None)
        else:
            train_texts, val_texts = comps["datasets"].get("webdataset")(data_cfg.get("path", ""), text_field, None)

    # Create TrainerConfig from dictionary
    cfg = TrainerConfig.from_dict(raw_cfg)
    
    # Package data options for trainer (collate, bucketing)
    data_options = {
        "collate": str(raw_cfg.get("data", {}).get("collate", "lm")),
        "bucket_by_length": bool(raw_cfg.get("data", {}).get("bucket_by_length", False)),
        "bucket_bins": list(raw_cfg.get("data", {}).get("bucket_bins", [64, 128, 256, 512])),
    }

    trainer = GenericTrainer(
        cfg=cfg,
        train_texts=train_texts,
        val_texts=val_texts,
        text_field_max_len=int(raw_cfg.get("data", {}).get("max_seq_len", max_seq_len)),
        callbacks=callbacks,
        data_options=data_options,
    )
    
    return trainer

