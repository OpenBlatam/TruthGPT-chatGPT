# Guía rápida: Modelos adaptables y rápidos en TruthGPT

Enfoque práctico para añadir nuevos modelos (LLMs y Diffusers) con alto rendimiento y mínima fricción.

## Índice

- Adaptabilidad con fábrica/registry de modelos
- Configuración por YAML (LLM y Difusión)
- Ejemplos mínimos: Transformers y Diffusers
- Entrenamiento eficiente (AMP, grad-accum, DDP)
- Checklist de aceleración y librerías recomendadas

## Adaptabilidad: fábrica y contrato de modelo

Define un contrato simple y un registry para descubrir modelos sin tocar el core.

```python
from typing import Protocol, Dict, Any, Callable

class InferenceModel(Protocol):
    def load(self, cfg: Dict[str, Any]) -> None: ...
    def infer(self, inputs: Dict[str, Any]) -> Dict[str, Any]: ...

MODEL_REGISTRY: Dict[str, Callable[[], InferenceModel]] = {}

def register_model(name: str):
    def _wrap(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return _wrap

def build_model(name: str, cfg: Dict[str, Any]) -> InferenceModel:
    model = MODEL_REGISTRY[name]()
    model.load(cfg)
    return model
```

## Configuración YAML unificada

```yaml
run_name: demo
seed: 42
task: text-generation  # o image-generation
model:
  family: hf-transformers  # hf-transformers | hf-diffusers | custom
  name_or_path: gpt2
  revision: null
training:
  epochs: 3
  train_batch_size: 8
  eval_batch_size: 8
  grad_accum_steps: 2
  learning_rate: 5.0e-5
  scheduler: cosine
  warmup_ratio: 0.06
  mixed_precision: bf16  # fp16|bf16|no
  max_grad_norm: 1.0
  early_stopping_patience: 2
model_optim:
  gradient_checkpointing: true
  lora:
    enabled: false
    r: 16
    alpha: 32
    dropout: 0.05
hardware:
  device: auto
data:
  dataset: wikitext
  subset: wikitext-2-raw-v1
  text_field: text
  max_seq_len: 512
```

## Ejemplo mínimo: LLM con Transformers

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

@register_model("hf-transformers")
class HFLLM:
    def __init__(self):
        self.tokenizer = None
        self.model = None

    def load(self, cfg):
        name = cfg["model"]["name_or_path"]
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16 if cfg["training"]["mixed_precision"]=="bf16" else None)
        if cfg.get("model_optim", {}).get("gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable()
        self.model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

    @torch.inference_mode()
    def infer(self, inputs):
        device = next(self.model.parameters()).device
        toks = self.tokenizer(inputs["text"], return_tensors="pt").to(device)
        out = self.model.generate(**toks, max_new_tokens=inputs.get("max_new_tokens", 64))
        return {"text": self.tokenizer.decode(out[0], skip_special_tokens=True)}
```

## Ejemplo mínimo: Difusión con Diffusers

```python
import torch
from diffusers import StableDiffusionPipeline

@register_model("hf-diffusers")
class HFDiffusion:
    def __init__(self):
        self.pipe = None

    def load(self, cfg):
        name = cfg["model"]["name_or_path"]
        self.pipe = StableDiffusionPipeline.from_pretrained(name, torch_dtype=torch.float16)
        self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe.enable_attention_slicing()
        self.pipe.enable_model_cpu_offload() if not torch.cuda.is_available() else None

    @torch.inference_mode()
    def infer(self, inputs):
        image = self.pipe(inputs["prompt"], num_inference_steps=inputs.get("steps", 25)).images[0]
        return {"image": image}
```

## Entrenamiento eficiente (AMP, grad-accum, DDP)

```python
import math, torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

def train_step(model, batch, cfg):
    device = next(model.parameters()).device
    scaler = GradScaler(enabled=cfg["training"]["mixed_precision"] in {"fp16", "bf16"})
    optimizer = AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=0.01)
    model.train()
    optimizer.zero_grad()
    for step in range(cfg["training"]["grad_accum_steps"]):
        with autocast(enabled=cfg["training"]["mixed_precision"]!="no"):
            outputs = model(**{k:v.to(device) for k,v in batch.items()})
            loss = outputs.loss / cfg["training"]["grad_accum_steps"]
        scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["max_grad_norm"]) 
    scaler.step(optimizer)
    scaler.update()
```

Para multi‑GPU, usa DistributedDataParallel (DDP) o `accelerate` (HuggingFace) para simplificar.

## Checklist de rendimiento (más rápido)

- Mixed precision: fp16/bf16 en entrenamiento e inferencia.
- KV cache y `use_cache=True` en generación; `max_new_tokens` acotado.
- Gradient checkpointing para ahorrar memoria; aumenta batch global.
- Fusión de kernels/FlashAttention 2 si está disponible (PyTorch 2.x + CUDA >= 11.8).
- Compilación: `torch.compile(mode="max-autotune")` cuando sea estable para tu backend.
- Quantización de inferencia: bitsandbytes 8‑bit/4‑bit o `torch.ao.quantization`/`onnxruntime`/`TensorRT`.
- DataLoader: `num_workers>0`, `prefetch_factor`, `pin_memory=True`.

## Mejores librerías por tarea

- LLMs: `transformers`, `datasets`, `accelerate`, `peft` (LoRA), `bitsandbytes` (8/4‑bit), `trl` (RLHF/ppo/dpo), `flash-attn`.
- Difusión: `diffusers`, `xformers` (mem/attn), `onnxruntime`/`TensorRT` para despliegue.
- Entrenamiento/monitoring: `wandb` o `tensorboard`, `tqdm`.
- Despliegue: `vLLM`/`tgi` (text), `FastAPI` + `onnxruntime` (vision/audio), `gradio` para demos.

## Ejecución: construir por nombre de modelo

```python
import yaml

def run_infer(cfg_path: str, inputs: dict):
    cfg = yaml.safe_load(open(cfg_path))
    name = cfg["model"]["family"]
    model = build_model(name, cfg)
    return model.infer(inputs)
```

Con este patrón, añadir un nuevo modelo implica: crear clase que implemente `load/infer`, decorarla con `@register_model("mi-modelo")` y referenciarla en YAML.


