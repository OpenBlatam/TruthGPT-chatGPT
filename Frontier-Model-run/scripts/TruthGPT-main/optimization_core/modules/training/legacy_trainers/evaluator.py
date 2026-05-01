"""
Evaluator — Handles model evaluation with Pydantic-validated results.

Separated from trainer for better modularity. Returns typed
``EvaluationResult`` with timing, batch stats, and computed metrics.
"""
import math
import time
import logging
from enum import Enum
from typing import Dict, Optional, Any

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from pydantic import BaseModel, Field, ConfigDict, computed_field

from optimization_core.trainers.config import TrainingConfig


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class MetricStrategy(str, Enum):
    """Strategy for selecting the best checkpoint metric."""
    LOSS = "loss"
    PERPLEXITY = "ppl"
    BITS_PER_BYTE = "bpb"


class EvaluationMetrics(BaseModel):
    """Pydantic validated evaluation metrics structure."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    loss: float = Field(..., description="Validation loss")
    perplexity: float = Field(..., description="Validation perplexity")
    additional_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Other monitored metrics",
    )

    @computed_field  # type: ignore[misc]
    @property
    def bits_per_byte(self) -> float:
        """Bits-per-byte = loss / ln(2). Lower is better."""
        if self.loss == float("inf") or math.isnan(self.loss):
            return float("inf")
        return round(self.loss / math.log(2), 4)


class EvaluationResult(BaseModel):
    """Complete evaluation output with timing and batch statistics."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    metrics: EvaluationMetrics
    batches_processed: int = 0
    batches_skipped: int = 0
    elapsed_ms: float = 0.0
    used_ema: bool = False

    @computed_field  # type: ignore[misc]
    @property
    def success_rate(self) -> float:
        total = self.batches_processed + self.batches_skipped
        if total == 0:
            return 0.0
        return round(self.batches_processed / total, 4)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Handles model evaluation.

    Responsibilities:
    - Evaluate model on validation set
    - Calculate metrics (loss, perplexity, bits-per-byte)
    - Support EMA weights during evaluation
    - Return typed ``EvaluationResult`` with timing
    """

    def __init__(
        self,
        training_config: TrainingConfig,
        model: torch.nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        use_amp: bool,
        ema_manager: Optional[object] = None,
    ):
        self.training_config = training_config
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.use_amp = use_amp
        self.ema_manager = ema_manager

    def _get_amp_dtype(self) -> Optional[torch.dtype]:
        """Get AMP dtype."""
        if self.training_config.mixed_precision == "bf16":
            return torch.bfloat16
        if self.training_config.mixed_precision == "fp16":
            return torch.float16
        return None

    @torch.no_grad()
    def evaluate(self) -> EvaluationResult:
        """
        Evaluate model on validation set.

        Returns:
            EvaluationResult containing metrics, timing, and batch stats.
        """
        start = time.monotonic()
        used_ema = False

        # Apply EMA weights if available
        if self.ema_manager and self.ema_manager.ema_config.enabled:
            self.ema_manager.apply_ema()
            used_ema = True

        self.model.eval()
        total_loss = 0.0
        count = 0
        skipped = 0

        try:
            for batch in self.val_loader:
                try:
                    batch = {
                        k: v.to(self.device, non_blocking=True)
                        for k, v in batch.items()
                    }

                    with autocast(
                        device_type=str(self.device.type),
                        enabled=self.use_amp,
                        dtype=self._get_amp_dtype(),
                    ):
                        outputs = self.model(**batch)
                        loss = outputs.loss

                        if isinstance(loss, dict):
                            loss = loss.get("loss", list(loss.values())[0])
                        elif hasattr(loss, "mean"):
                            loss = loss.mean()

                    if torch.isfinite(loss):
                        total_loss += float(loss.detach().item())
                        count += 1
                    else:
                        logger.warning("Non-finite loss encountered: %s", loss.item())
                        skipped += 1

                except Exception as e:
                    logger.error("Error in evaluation batch: %s", e, exc_info=True)
                    skipped += 1
                    continue
        finally:
            self.model.train()
            if self.ema_manager and self.ema_manager.ema_config.enabled:
                self.ema_manager.restore_from_ema()

        elapsed_ms = (time.monotonic() - start) * 1000

        if count == 0:
            logger.warning("No valid evaluation samples processed")
            metrics = EvaluationMetrics(loss=float("inf"), perplexity=float("inf"))
        else:
            avg_loss = total_loss / count
            perplexity = (
                math.exp(min(20.0, max(-20.0, avg_loss)))
                if avg_loss == avg_loss
                else float("inf")
            )
            metrics = EvaluationMetrics(loss=avg_loss, perplexity=perplexity)

        logger.debug(
            "Evaluation: loss=%.4f, ppl=%.2f, bpb=%.4f, batches=%d/%d, %.1fms",
            metrics.loss, metrics.perplexity, metrics.bits_per_byte,
            count, count + skipped, elapsed_ms,
        )

        return EvaluationResult(
            metrics=metrics,
            batches_processed=count,
            batches_skipped=skipped,
            elapsed_ms=round(elapsed_ms, 2),
            used_ema=used_ema,
        )

    def select_best_metric(
        self, result: EvaluationResult, strategy: Optional[MetricStrategy] = None,
    ) -> float:
        """
        Select the metric to use for best-checkpoint selection.

        Args:
            result: EvaluationResult from evaluate()
            strategy: Override selection strategy (defaults to training_config)

        Returns:
            The selected metric value (lower is better).
        """
        strat = strategy or MetricStrategy(
            self.training_config.select_best_by
            if hasattr(self.training_config, "select_best_by")
            else "loss"
        )
        if strat == MetricStrategy.PERPLEXITY:
            return result.metrics.perplexity
        if strat == MetricStrategy.BITS_PER_BYTE:
            return result.metrics.bits_per_byte
        return result.metrics.loss

