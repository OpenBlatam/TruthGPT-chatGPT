from typing import Optional

from pydantic import BaseModel, Field


class TrainingCfg(BaseModel):
    epochs: int = Field(3, ge=1)
    train_batch_size: int = Field(8, ge=1)
    eval_batch_size: int = Field(8, ge=1)
    grad_accum_steps: int = Field(2, ge=1)
    learning_rate: float = Field(5e-5, gt=0)
    warmup_ratio: float = Field(0.06, ge=0, le=1)
    mixed_precision: str = Field("bf16")  # no|fp16|bf16


class ModelCfg(BaseModel):
    family: str = Field(...)
    name_or_path: str = Field(...)


class AppCfg(BaseModel):
    run_name: str = Field("run")
    seed: int = Field(42)
    model: ModelCfg
    training: TrainingCfg
    data: Optional[dict] = None






