"""
Main System Orchestrator
"""

from .imports import *
from .core import BaseModule
from .config_manager import ConfigManager
from .models import create_model_module
from .data import create_data_module
from .training import create_training_module
from .optimization import create_optimization_module
from .evaluation import create_evaluation_module
from .inference import create_inference_module
from .monitoring import create_monitoring_module

class ModularSystem:
    """Main modular system orchestrator"""
    
    def __init__(self, config_source: Union[str, Dict[str, Any]]):
        self.config_manager = ConfigManager(config_source)
        self.config = self.config_manager.config
        self.modules = {}
        self.devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._setup_modules()
    
    def _setup_modules(self):
        """Setup all modules"""
        # Model module
        model_type = self.config.get("model", {}).get("type", "transformer")
        model_config = self.config.get("model", {})
        self.modules["model"] = create_model_module(model_type, model_config)
        
        # Data module
        data_type = self.config.get("data", {}).get("type", "text")
        data_config = self.config.get("data", {})
        self.modules["data"] = create_data_module(data_type, data_config)
        
        # Training module
        training_type = self.config.get("training", {}).get("type", "supervised")
        training_config = self.config.get("training", {})
        self.modules["training"] = create_training_module(training_type, training_config)
        
        # Optimization module
        optimization_type = self.config.get("optimization", {}).get("type", "adamw")
        optimization_config = self.config.get("optimization", {})
        self.modules["optimization"] = create_optimization_module(optimization_type, optimization_config)
        
        # Evaluation module
        evaluation_type = self.config.get("evaluation", {}).get("type", "classification")
        evaluation_config = self.config.get("evaluation", {})
        self.modules["evaluation"] = create_evaluation_module(evaluation_type, evaluation_config)
        
        # Inference module
        inference_type = self.config.get("inference", {}).get("type", "text")
        inference_config = self.config.get("inference", {})
        self.modules["inference"] = create_inference_module(inference_type, inference_config)
        
        # Monitoring module
        monitoring_type = self.config.get("monitoring", {}).get("type", "performance")
        monitoring_config = self.config.get("monitoring", {})
        self.modules["monitoring"] = create_monitoring_module(monitoring_type, monitoring_config)
    
    def train(self, epochs: int):
        """Train the system"""
        self.modules["training"].train(
            self.modules["model"],
            self.modules["data"],
            epochs
        )
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the system"""
        return self.modules["evaluation"].evaluate(
            self.modules["model"].model,
            self.modules["data"].get_dataloader()
        )
    
    def predict(self, input_data: Any) -> Any:
        """Make prediction"""
        return self.modules["inference"].predict(input_data)
    
    def monitor(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor the system"""
        return self.modules["monitoring"].monitor(data)
    
    def get_module(self, module_name: str) -> BaseModule:
        """Get specific module"""
        return self.modules.get(module_name)
    
    def update_config(self, key: str, value: Any):
        """Update configuration"""
        self.config_manager.update(key, value)
        self.config_manager.save()
    
    def save_checkpoint(self, path: str):
        """Save system checkpoint"""
        checkpoint = {
            "config": self.config,
            "model_state": self.modules["model"].model.state_dict(),
            "optimizer_state": self.modules["model"].optimizer.state_dict(),
            "scheduler_state": self.modules["model"].scheduler.state_dict() if self.modules["model"].scheduler else None
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load system checkpoint"""
        checkpoint = torch.load(path, map_location=self.devices)
        self.modules["model"].model.load_state_dict(checkpoint["model_state"])
        self.modules["model"].optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint["scheduler_state"] and self.modules["model"].scheduler:
            self.modules["model"].scheduler.load_state_dict(checkpoint["scheduler_state"])

