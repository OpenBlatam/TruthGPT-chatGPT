"""
Model Modules
"""

from .imports import *
from .core import BaseModule

class ModelModule(BaseModule):
    """Base class for model modules"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
    
    def _setup(self):
        """Setup model components"""
        self._create_model()
        self._create_optimizer()
        self._create_scheduler()
        if self.config.get("use_mixed_precision", False):
            self.scaler = amp.GradScaler()
    
    @abstractmethod
    def _create_model(self):
        """Create the model"""
        pass
    
    def _create_optimizer(self):
        """Create optimizer"""
        optimizer_type = self.config.get("optimizer", "adamw")
        lr = self.config.get("learning_rate", 1e-4)
        
        if self.model is None:
            self.logger.warning("Model is None, skipping optimizer creation")
            return

        if optimizer_type == "adamw":
            self.optimizer = AdamW(self.model.parameters(), lr=lr)
        elif optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.optimizer is None:
            return

        scheduler_type = self.config.get("scheduler", "cosine")
        
        if scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.get("warmup_steps", 100),
                num_training_steps=self.config.get("total_steps", 1000)
            )
        elif scheduler_type == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.get("warmup_steps", 100),
                num_training_steps=self.config.get("total_steps", 1000)
            )
        else:
            self.scheduler = None
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        if self.model is None or self.optimizer is None:
            return {}

        self.model.train()
        self.optimizer.zero_grad()
        
        if self.scaler:
            with amp.autocast():
                outputs = self.model(**batch)
                loss = outputs.loss
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
        
        if self.scheduler:
            self.scheduler.step()
        
        return {"loss": loss.item()}
    
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single evaluation step"""
        if self.model is None:
            return {}

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**batch)
            loss = outputs.loss
        return {"loss": loss.item()}

class TransformerModule(ModelModule):
    """Transformer model module"""
    
    def _create_model(self):
        """Create transformer model"""
        model_name = self.config.get("model_name", "bert-base-uncased")
        self.model = AutoModel.from_pretrained(model_name)
        
        # Add custom head if specified
        if self.config.get("add_classification_head", False):
            num_labels = self.config.get("num_labels", 2)
            self.model.classifier = nn.Linear(self.model.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass"""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        if hasattr(self.model, 'classifier'):
            logits = self.model.classifier(outputs.last_hidden_state[:, 0])
        else:
            logits = outputs.last_hidden_state[:, 0]
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return type('Outputs', (), {'logits': logits, 'loss': loss})()

class DiffusionModule(ModelModule):
    """Diffusion model module"""
    
    def _create_model(self):
        """Create diffusion model"""
        model_name = self.config.get("model_name", "runwayml/stable-diffusion-v1-5")
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.config.get("use_fp16", False) else torch.float32
        )
        self.model = self.pipeline.unet
    
    def forward(self, latents, timestep, encoder_hidden_states):
        """Forward pass"""
        return self.model(latents, timestep, encoder_hidden_states)
    
    def generate(self, prompt: str, num_images: int = 1, **kwargs):
        """Generate images"""
        return self.pipeline(prompt, num_images_per_prompt=num_images, **kwargs)

class LoRAModule(ModelModule):
    """LoRA (Low-Rank Adaptation) module"""
    
    def _create_model(self):
        """Create base model with LoRA"""
        base_model_name = self.config.get("base_model", "bert-base-uncased")
        self.base_model = AutoModel.from_pretrained(base_model_name)
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.get("lora_r", 8),
            lora_alpha=self.config.get("lora_alpha", 32),
            lora_dropout=self.config.get("lora_dropout", 0.1),
            target_modules=self.config.get("target_modules", ["q_proj", "v_proj"])
        )
        
        self.model = get_peft_model(self.base_model, lora_config)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass"""
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

class QuantizedModule(ModelModule):
    """Quantized model module"""
    
    def _create_model(self):
        """Create quantized model"""
        model_name = self.config.get("model_name", "bert-base-uncased")
        
        # Configure quantization
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=self.config.get("load_in_8bit", True),
            load_in_4bit=self.config.get("load_in_4bit", False),
            llm_int8_threshold=self.config.get("llm_int8_threshold", 6.0)
        )
        
        self.model = AutoModel.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass"""
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

def create_model_module(model_type: str, config: Dict[str, Any]) -> ModelModule:
    """Create model module"""
    if model_type == "transformer":
        return TransformerModule(config)
    elif model_type == "diffusion":
        return DiffusionModule(config)
    elif model_type == "lora":
        return LoRAModule(config)
    elif model_type == "quantized":
        return QuantizedModule(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

