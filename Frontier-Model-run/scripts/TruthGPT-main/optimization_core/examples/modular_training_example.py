"""
Example demonstrating the complete modular architecture.
Shows how to use services, events, and plugins together.
"""
import torch
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.cuda.amp import GradScaler

# Core modules
from modules.base.core_system.core.config import ConfigManager
from modules.base.core_system.core.service_registry import ServiceContainer
from modules.base.core_system.core.event_system import EventType, on_event
from modules.base.core_system.core.services import ModelService, TrainingService

# Data modules
from data import DatasetManager, DataLoaderFactory

# Training modules
from training.experiment_tracker import ExperimentTracker

# Utils
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def setup_training(config_path: str):
    """
    Complete training setup using modular architecture.
    
    Args:
        config_path: Path to YAML configuration
    """
    # 1. Load configuration
    config = ConfigManager.load_config(config_path)
    logger.info(f"Configuration loaded: {config.run_name}")
    
    # 2. Setup event handlers
    def on_training_step(event):
        step = event.data.get("step", 0)
        loss = event.data.get("loss", 0.0)
        if step % config.training.log_interval == 0:
            logger.info(f"Step {step}: loss={loss:.4f}")
    
    def on_checkpoint_saved(event):
        logger.info(f"Checkpoint saved: {event.data['path']}")
    
    on_event(EventType.TRAINING_STEP, on_training_step)
    on_event(EventType.CHECKPOINT_SAVED, on_checkpoint_saved)
    
    # 3. Create service container
    container = ServiceContainer()
    
    # 4. Register and initialize services
    model_service = ModelService()
    model_service.initialize()
    container.register("model_service", model_service, singleton=True)
    
    training_service = TrainingService()
    training_service.initialize()
    container.register("training_service", training_service, singleton=True)
    
    # 5. Load model
    logger.info("Loading model...")
    model = model_service.load_model(
        model_name=config.model.name_or_path,
        config={
            "torch_dtype": torch.bfloat16 if config.training.mixed_precision == "bf16" else None,
            "gradient_checkpointing": config.model.gradient_checkpointing,
            "lora": {
                "enabled": config.model.lora_enabled,
                "r": config.model.lora_r,
                "alpha": config.model.lora_alpha,
                "dropout": config.model.lora_dropout,
            } if config.model.lora_enabled else None,
        }
    )
    
    # 6. Load data
    logger.info("Loading data...")
    train_texts, val_texts = DatasetManager.load_dataset(
        source=config.data.source,
        dataset_name=config.data.dataset,
        subset=config.data.subset,
        text_field=config.data.text_field,
        streaming=config.data.streaming,
    )
    
    # 7. Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 8. Create data loaders
    train_loader = DataLoaderFactory.create_train_loader(
        texts=train_texts,
        tokenizer=tokenizer,
        max_length=config.data.max_seq_len,
        batch_size=config.training.train_batch_size,
        collate_type=config.data.collate,
        bucket_by_length=config.data.bucket_by_length,
        bucket_bins=config.data.bucket_bins,
        num_workers=config.data.num_workers,
        prefetch_factor=config.data.prefetch_factor,
        persistent_workers=config.data.persistent_workers,
    )
    
    val_loader = DataLoaderFactory.create_val_loader(
        texts=val_texts,
        tokenizer=tokenizer,
        max_length=config.data.max_seq_len,
        batch_size=config.training.eval_batch_size,
        collate_type=config.data.collate,
        num_workers=config.data.num_workers,
        prefetch_factor=config.data.prefetch_factor,
        persistent_workers=config.data.persistent_workers,
    )
    
    # 9. Setup training components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Multi-GPU if configured
    if config.hardware.multi_gpu and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
    
    # Optimizer
    from factories.optimizer import OPTIMIZERS
    optimizer = OPTIMIZERS.build(
        config.training.optimizer_type,
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        fused=config.training.fused_adamw,
    )
    
    # Scheduler
    num_train_steps = len(train_loader) * config.training.epochs // config.training.grad_accum_steps
    num_warmup = int(config.training.warmup_ratio * num_train_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup,
        num_training_steps=num_train_steps,
    )
    
    # Scaler
    use_amp = config.training.mixed_precision != "none"
    scaler = GradScaler(enabled=use_amp)
    
    # 10. Configure training service
    training_service.configure(
        config={
            "use_amp": use_amp,
            "mixed_precision": config.training.mixed_precision,
            "max_grad_norm": config.training.max_grad_norm,
            "grad_accum_steps": config.training.grad_accum_steps,
            "ema_enabled": config.ema.enabled,
            "ema_decay": config.ema.decay,
            "tracking_enabled": bool(config.logging),
            "trackers": config.training.callbacks,
            "project": config.logging.get("project"),
            "run_name": config.logging.get("run_name", config.run_name),
            "log_dir": config.logging.get("dir"),
        },
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        output_dir=config.output_dir,
    )
    
    # 11. Training loop
    logger.info("Starting training...")
    best_val_loss = float("inf")
    patience_counter = 0
    
    training_service.emit(EventType.TRAINING_STARTED, {
        "epochs": config.training.epochs,
        "total_steps": num_train_steps,
    })
    
    for epoch in range(config.training.epochs):
        # Train epoch
        epoch_metrics = training_service.train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
        )
        
        # Evaluate
        if (epoch + 1) % (config.training.eval_interval // len(train_loader)) == 0:
            val_metrics = training_service.evaluate(
                model=model,
                val_loader=val_loader,
                device=device,
            )
            
            logger.info(
                f"Epoch {epoch + 1}/{config.training.epochs} | "
                f"Train Loss: {epoch_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Perplexity: {val_metrics['perplexity']:.2f}"
            )
            
            # Check for improvement
            current_loss = val_metrics["loss"]
            if current_loss < best_val_loss:
                best_val_loss = current_loss
                patience_counter = 0
                
                # Save best checkpoint
                checkpoint_path = f"{config.output_dir}/best.pt"
                training_service.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    step=epoch,
                    path=checkpoint_path,
                    tokenizer=tokenizer,
                )
                logger.info(f"New best model saved (loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                
                # Early stopping
                if patience_counter >= config.training.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Periodic checkpoint
        if (epoch + 1) % config.checkpoint.interval_steps == 0:
            checkpoint_path = f"{config.output_dir}/checkpoint_epoch_{epoch + 1}"
            training_service.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                step=epoch,
                path=checkpoint_path,
                tokenizer=tokenizer,
            )
    
    # Save final checkpoint
    training_service.save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        step=config.training.epochs,
        path=f"{config.output_dir}/final.pt",
        tokenizer=tokenizer,
    )
    
    training_service.finish()
    logger.info("Training completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Modular training example")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/llm_default.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    try:
        setup_training(args.config)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise



