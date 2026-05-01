import torch
from diffusers import StableDiffusionPipeline

from . import register_model


@register_model("hf-diffusers")
class HFDiffusion:
    def __init__(self) -> None:
        self.pipe = None

    def load(self, cfg):
        name = cfg["model"]["name_or_path"]
        dtype = torch.float16 if torch.cuda.is_available() else None
        self.pipe = StableDiffusionPipeline.from_pretrained(name, torch_dtype=dtype)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe.to(device)
        try:
            self.pipe.enable_attention_slicing()
        except Exception:
            pass
        if not torch.cuda.is_available():
            try:
                self.pipe.enable_model_cpu_offload()
            except Exception:
                pass

    @torch.inference_mode()
    def infer(self, inputs):
        prompt = inputs.get("prompt") or inputs.get("text") or ""
        steps = int(inputs.get("steps", 25))
        images = self.pipe(prompt, num_inference_steps=steps).images
        return {"image": images[0]}






