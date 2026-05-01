"""
Enhanced GenericTrainer with best practices for PyTorch, Transformers, and LLM training.

Follows best practices for:
- Mixed precision training (FP16/BF16)
- Gradient accumulation and clipping
- Weight initialization and normalization
- Error handling and NaN detection
- Multi-GPU support (DataParallel/DDP)
- Efficient data loading with dynamic padding
- EMA (Exponential Moving Average) weights
- Comprehensive logging and monitoring
"""
import math
import os
import random
from typing import Dict, Optional, List, Any, Union, Tuple
import logging
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from pydantic import BaseModel
# from optimization_core.adapters import create_adapter, ObjectStore
from optimization_core.trainers.model_manager import ModelManager, ParallelMode
from optimization_core.trainers.data_manager import DataManager
from optimization_core.trainers.optimizer_manager import OptimizerManager
from optimization_core.trainers.ema_manager import EMAManager
from optimization_core.trainers.checkpoint_manager import CheckpointManager

from optimization_core.trainers.config import TrainerConfig, TrainingConfig
from optimization_core.trainers.evaluator import Evaluator, EvaluationResult, MetricStrategy
from optimization_core.trainers.callbacks import Callback
from optimization_core.utils.logging_utils import TrainingLogger

try:
    import torch.distributed as dist
    _DISTRIBUTED_AVAILABLE = True
except ImportError:
    _DISTRIBUTED_AVAILABLE = False

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class GenericTrainer:
    def __init__(
        self,
        cfg: TrainerConfig,
        train_texts: Optional[List[str]] = None,
        val_texts: Optional[List[str]] = None,
        text_field_max_len: int = 512,
        callbacks: Optional[List[Callback]] = None,
        data_options: Optional[Dict[str, Any]] = None,
        model_id: Optional[str] = None,
        data_id: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        set_seed(cfg.seed)
        self.callbacks = callbacks or []
        self.data_options = data_options or {}
        self.store = ObjectStore.instance()

        # 1. Hardware/Device setup
        self.device = self._resolve_device(cfg.hardware.device)

        # 2. Model & Tokenizer Handling
        self.model_manager = ModelManager(
            model_config=cfg.model,
            hardware_config=cfg.hardware,
            training_config=cfg.training,
            device=self.device,
        )

        if model_id:
            self.model_id = model_id
            self.model = self.store.get(model_id)
            # Try to get tokenizer from Store if it was registered
            self.tokenizer = self.store.get_kind("tokenizer", first=True) 
            if not self.tokenizer:
                self.tokenizer = self.model_manager.load_tokenizer()
        else:
            self.tokenizer = self.model_manager.load_tokenizer()
            self.model, load_result = self.model_manager.load_model()
            
            # Register in ObjectStore for dynamic tool usage
            self.model_id = self.store.put(
                self.model, 
                kind="model", 
                meta={"path": cfg.model.name_or_path}
            )
        
        logger.info("GenericTrainer: Model registered as %s", self.model_id)

        # Enable TF32 where beneficial
        if self.device.type == "cuda" and cfg.hardware.allow_tf32:
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        # 3. Data Handling
        if data_id:
            dataset = self.store.get(data_id)
            if isinstance(dataset, dict):
                train_texts = dataset.get("train", [])
                val_texts = dataset.get("val", [])
            else:
                train_texts = dataset
                val_texts = []
        
        self.data_manager = DataManager(
            training_config=cfg.training,
            hardware_config=cfg.hardware,
            tokenizer=self.tokenizer,
            text_field_max_len=text_field_max_len,
            data_options=self.data_options,
        )
        self.train_loader, self.val_loader, data_res = self.data_manager.create_loaders(
            train_texts if train_texts is not None else [],
            val_texts if val_texts is not None else []
        )
        logger.info("DataManager: %s", data_res.model_dump_json())

        # 4. Optimizer & Scheduler
        self.optimizer_manager = OptimizerManager(
            training_config=self.cfg.training,
            model=self.model,
            use_amp=self._use_amp(),
        )
        
        # Calculate total steps for scheduler
        num_train_steps = max(1, (len(self.train_loader or []) * self.cfg.training.epochs) // max(1, self.cfg.training.grad_accum_steps))
        
        # Unified setup
        opt_res = self.optimizer_manager.setup(num_training_steps=num_train_steps)
        self.optimizer = self.optimizer_manager.optimizer
        self.lr_scheduler = self.optimizer_manager.scheduler
        self.scaler = self.optimizer_manager.scaler
        
        logger.info("GenericTrainer: Optimizer setup complete: %s", opt_res.model_dump_json())

        # 5. EMA via EMAManager
        self.ema_manager = EMAManager(ema_config=self.cfg.training.ema, model=self.model)

        # 6. Checkpoints via CheckpointManager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_config=self.cfg.checkpoint,
            output_dir=self.cfg.training.output_dir,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            scaler=self.scaler,
            tokenizer=self.tokenizer,
        )

        # 7. Multi-GPU setup
        self._is_parallel = False
        self._is_ddp = False
        if cfg.hardware.ddp and _DISTRIBUTED_AVAILABLE:
            self.model_manager.setup_parallel(ParallelMode.DDP)
            self.model = self.model_manager.model
            self.device = self.model_manager.device
            self._is_ddp = True
        elif cfg.hardware.multi_gpu and torch.cuda.device_count() > 1:
            self.model_manager.setup_parallel(ParallelMode.DATA_PARALLEL)
            self.model = self.model_manager.model
            self._is_parallel = True

        self.training_logger = TrainingLogger(logger)

        # 8. Evaluator Proxy
        class TrainerEMAManagerProxy:
            def __init__(self, manager):
                self.manager = manager
            def apply_ema(self):
                self.manager.apply_ema()
            def restore_from_ema(self):
                self.manager.restore_from_ema()

        ema_proxy = TrainerEMAManagerProxy(self.ema_manager) if self.ema_manager else None

        self.evaluator = Evaluator(
            training_config=self.cfg.training,
            model=self.model,
            val_loader=self.val_loader,
            device=self.device,
            use_amp=self._use_amp(),
            ema_manager=ema_proxy,
        )

    def _resolve_device(self, target: str) -> torch.device:
        if target == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(target)

    def _use_amp(self) -> bool:
        return self.device.type == "cuda" and self.cfg.training.mixed_precision in ("fp16", "bf16")

    def _amp_dtype(self):
        if self.cfg.training.mixed_precision == "bf16":
            return torch.bfloat16
        return torch.float16

    def train(self) -> None:
        """Main training loop."""
        best_val = math.inf
        best_metric = math.inf
        bad_epochs = 0
        global_step = 0
        
        # Resume from checkpoint
        if self.cfg.checkpoint.resume_from_checkpoint:
            resume_state = self.checkpoint_manager.load(self.cfg.checkpoint.resume_from_checkpoint)
            global_step = resume_state.get("step", 0)
            self.training_logger.log_info(f"Resumed training from step {global_step}")
        
        self.model.train()
        
        try:
            for epoch in range(self.cfg.training.epochs):
                running_loss = 0.0
                epoch_start = time.perf_counter()
                tokens_accum = 0
                step_count = 0
                
                logger.info(f"Starting epoch {epoch + 1}/{self.cfg.training.epochs}")
                
                for step, batch in enumerate(self.train_loader, start=1):
                    try:
                        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

                        with autocast(device_type=self.device.type, enabled=self._use_amp(), dtype=self._amp_dtype()):
                            outputs = self.model(**batch)
                            loss = outputs.loss / self.cfg.training.grad_accum_steps
                            if hasattr(loss, "mean"):
                                loss = loss.mean()

                        if not torch.isfinite(loss):
                            logger.warning(f"Non-finite loss at step {step}: {loss.item()}. Skipping.")
                            if self.optimizer:
                                self.optimizer.zero_grad(set_to_none=True)
                            continue
                        
                        if self.scaler:
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()
                        
                        if step % self.cfg.training.grad_accum_steps == 0:
                            if self.scaler and self.optimizer:
                                self.scaler.unscale_(self.optimizer)
                            
                            # Gradient clipping
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.cfg.training.max_grad_norm,
                            )
                            
                            # Optimizer step
                            self.optimizer_manager.step()
                            
                            # Update EMA
                            if self.ema_manager:
                                self.ema_manager.update()
                            
                            if self.optimizer:
                                self.optimizer.zero_grad(set_to_none=True)
                            self.optimizer_manager.scheduler_step()
                            
                            global_step += 1
                            step_count += 1

                        running_loss += loss.detach().item()
                        if "attention_mask" in batch:
                            tokens_accum += int(batch["attention_mask"].sum().item())

                        # Logging
                        if global_step and global_step % self.cfg.training.log_interval == 0:
                            avg_loss = running_loss / max(1, step_count)
                            elapsed = max(1e-6, time.perf_counter() - epoch_start)
                            tps = tokens_accum / elapsed
                            current_lr = self.optimizer_manager.get_lr()
                            
                            self.training_logger.log_step(
                                step=global_step, epoch=epoch + 1,
                                loss=avg_loss, learning_rate=current_lr, tokens_per_sec=tps
                            )
                            
                            running_loss = 0.0
                            tokens_accum = 0
                            step_count = 0
                            epoch_start = time.perf_counter()

                        # Evaluation
                        if global_step and global_step % self.cfg.training.eval_interval == 0:
                            eval_result = self.evaluator.evaluate()
                            val_loss = eval_result.metrics.loss
                            metric_value = self.evaluator.select_best_metric(eval_result)
                            improved = metric_value < (best_metric if self.cfg.training.select_best_by == "ppl" else best_val)
                            
                            self.training_logger.log_eval(
                                step=global_step, val_loss=val_loss,
                                perplexity=eval_result.metrics.perplexity, improved=improved
                            )
                            
                            if improved:
                                best_val = val_loss
                                best_metric = metric_value
                                bad_epochs = 0
                                self.checkpoint_manager.save("best.pt", step=global_step, is_best=True)
                            else:
                                bad_epochs += 1
                                if bad_epochs >= self.cfg.training.early_stopping_patience:
                                    logger.info("Early stopping triggered")
                                    self.checkpoint_manager.save("last.pt", step=global_step)
                                    return
                        
                        # Checkpoint interval
                        if global_step and (global_step % max(1, self.cfg.checkpoint.interval_steps) == 0):
                            self.checkpoint_manager.save(f"step_{global_step}.pt", step=global_step)
                            self.checkpoint_manager.prune_checkpoints()

                    except Exception as e:
                        logger.error(f"Error in training step {step}: {e}", exc_info=True)
                        if self.optimizer:
                            self.optimizer.zero_grad(set_to_none=True)
                        continue

            self.checkpoint_manager.save("last.pt", step=global_step)
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Training error: {e}", exc_info=True)
            self.checkpoint_manager.save("last.pt", step=global_step)
            raise

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.1
    ) -> str:
        """Generate text from prompt."""
        try:
            self.model.eval()
            model_for_gen = self.model_manager.get_model_for_operations()
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with autocast(device_type=self.device.type, enabled=self._use_amp(), dtype=self._amp_dtype()):
                output_ids = model_for_gen.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            self.model.train()
            return text
            
        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            self.model.train()
            raise

