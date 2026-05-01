"""
Inference Modules
"""

from .imports import *
from .core import BaseModule

class InferenceModule(BaseModule):
    """Base inference module"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
    
    def _setup(self):
        """Setup inference components"""
        self._load_model()
        self._load_tokenizer()
    
    @abstractmethod
    def _load_model(self):
        """Load model"""
        pass
    
    @abstractmethod
    def _load_tokenizer(self):
        """Load tokenizer"""
        pass
    
    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Make prediction"""
        pass

class TextInferenceModule(InferenceModule):
    """Text inference module"""
    
    def _load_model(self):
        """Load text model"""
        model_name = self.config.get("model_name", "bert-base-uncased")
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def _load_tokenizer(self):
        """Load tokenizer"""
        tokenizer_name = self.config.get("tokenizer_name", "bert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict on text"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return {
            "logits": outputs.last_hidden_state,
            "pooler_output": outputs.pooler_output
        }

class ImageInferenceModule(InferenceModule):
    """Image inference module"""
    
    def _load_model(self):
        """Load image model"""
        model_name = self.config.get("model_name", "resnet50")
        self.model = create_model(model_name, pretrained=True)
        self.model.eval()
    
    def _load_tokenizer(self):
        """Load image preprocessing"""
        self.tokenizer = None  # Images don't use tokenizers
    
    def predict(self, image: torch.Tensor) -> Dict[str, Any]:
        """Predict on image"""
        with torch.no_grad():
            outputs = self.model(image)
        
        return {
            "logits": outputs,
            "probabilities": F.softmax(outputs, dim=-1)
        }

class DiffusionInferenceModule(InferenceModule):
    """Diffusion inference module"""
    
    def _load_model(self):
        """Load diffusion model"""
        model_name = self.config.get("model_name", "runwayml/stable-diffusion-v1-5")
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.config.get("use_fp16", False) else torch.float32
        )
        self.model = self.pipeline.unet
    
    def _load_tokenizer(self):
        """Load text tokenizer for diffusion"""
        self.tokenizer = self.pipeline.tokenizer
    
    def predict(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate image from prompt"""
        images = self.pipeline(prompt, **kwargs)
        return {
            "images": images.images,
            "prompt": prompt
        }

def create_inference_module(inference_type: str, config: Dict[str, Any]) -> InferenceModule:
    """Create inference module"""
    if inference_type == "text":
        return TextInferenceModule(config)
    elif inference_type == "image":
        return ImageInferenceModule(config)
    elif inference_type == "diffusion":
        return DiffusionInferenceModule(config)
    else:
        raise ValueError(f"Unknown inference type: {inference_type}")

