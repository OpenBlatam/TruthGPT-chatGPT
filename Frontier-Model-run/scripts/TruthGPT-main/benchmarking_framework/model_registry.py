"""
Comprehensive model registry for benchmarking open source and closed source models
"""

import torch
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

class ModelType(Enum):
    OPEN_SOURCE = "open_source"
    CLOSED_SOURCE = "closed_source"
    TRUTHGPT = "truthgpt"

class ModelCategory(Enum):
    LANGUAGE_MODEL = "language_model"
    MULTIMODAL = "multimodal"
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    MATH = "math"
    VISION = "vision"

@dataclass
class ModelInfo:
    name: str
    model_type: ModelType
    category: ModelCategory
    parameters: int
    context_length: int
    provider: str
    api_endpoint: Optional[str] = None
    local_path: Optional[str] = None
    huggingface_id: Optional[str] = None
    description: str = ""
    capabilities: List[str] = None
    license: str = "unknown"
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []

class ModelRegistry:
    """Registry of top open source and closed source models for benchmarking."""
    
    def __init__(self):
        self.models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize registry with top models in each category."""
        
        self.register_model(ModelInfo(
            name="TruthGPT-DeepSeek-V3",
            model_type=ModelType.TRUTHGPT,
            category=ModelCategory.LANGUAGE_MODEL,
            parameters=1_550_312,
            context_length=2048,
            provider="TruthGPT",
            local_path="/home/ubuntu/TruthGPT/Frontier-Model-run/models/deepseek_v3.py",
            description="Native DeepSeek-V3 implementation with MLA and MoE optimizations",
            capabilities=["text_generation", "reasoning", "mcts_optimization"],
            license="MIT"
        ))
        
        self.register_model(ModelInfo(
            name="TruthGPT-Viral-Clipper",
            model_type=ModelType.TRUTHGPT,
            category=ModelCategory.MULTIMODAL,
            parameters=21_430_531,
            context_length=1024,
            provider="TruthGPT",
            local_path="/home/ubuntu/TruthGPT/variant/viral_clipper.py",
            description="Multi-modal viral video content detection and clipping",
            capabilities=["video_analysis", "viral_detection", "multimodal_fusion"],
            license="MIT"
        ))
        
        self.register_model(ModelInfo(
            name="TruthGPT-Brand-Analyzer",
            model_type=ModelType.TRUTHGPT,
            category=ModelCategory.MULTIMODAL,
            parameters=9_476_347,
            context_length=1024,
            provider="TruthGPT",
            local_path="/home/ubuntu/TruthGPT/brandkit/brand_analyzer.py",
            description="Website brand analysis and content generation",
            capabilities=["brand_analysis", "content_generation", "web_scraping"],
            license="MIT"
        ))
        
        self.register_model(ModelInfo(
            name="TruthGPT-Llama-3.1-405B",
            model_type=ModelType.TRUTHGPT,
            category=ModelCategory.LANGUAGE_MODEL,
            parameters=405_000_000_000,
            context_length=131072,
            provider="TruthGPT",
            local_path="/home/ubuntu/TruthGPT/Frontier-Model-run/models/llama_3_1_405b.py",
            description="Native Llama-3.1-405B implementation with GQA and scaled RoPE",
            capabilities=["text_generation", "reasoning", "code_generation", "long_context"],
            license="MIT"
        ))
        
        self.register_model(ModelInfo(
            name="TruthGPT-Claude-3.5-Sonnet",
            model_type=ModelType.TRUTHGPT,
            category=ModelCategory.LANGUAGE_MODEL,
            parameters=175_000_000_000,
            context_length=200000,
            provider="TruthGPT",
            local_path="/home/ubuntu/TruthGPT/Frontier-Model-run/models/claude_3_5_sonnet.py",
            description="Native Claude-3.5-Sonnet implementation with Constitutional AI",
            capabilities=["text_generation", "reasoning", "constitutional_ai", "safety"],
            license="MIT"
        ))
        
        self.register_model(ModelInfo(
            name="TruthGPT-Qwen-Optimized",
            model_type=ModelType.TRUTHGPT,
            category=ModelCategory.REASONING,
            parameters=1_243_820_544,
            context_length=32768,
            provider="TruthGPT",
            local_path="/home/ubuntu/TruthGPT/qwen_variant/qwen_model.py",
            description="Optimized Qwen model with advanced reasoning capabilities",
            capabilities=["reasoning", "math", "code_generation", "long_context"],
            license="MIT"
        ))
        
        self.register_model(ModelInfo(
            name="Llama-3.1-405B",
            model_type=ModelType.OPEN_SOURCE,
            category=ModelCategory.LANGUAGE_MODEL,
            parameters=405_000_000_000,
            context_length=128000,
            provider="Meta",
            huggingface_id="meta-llama/Llama-3.1-405B-Instruct",
            description="Meta's largest open source language model",
            capabilities=["text_generation", "reasoning", "code_generation", "multilingual"],
            license="Llama 3.1 Community License"
        ))
        
        self.register_model(ModelInfo(
            name="Qwen2.5-72B",
            model_type=ModelType.OPEN_SOURCE,
            category=ModelCategory.LANGUAGE_MODEL,
            parameters=72_000_000_000,
            context_length=32768,
            provider="Alibaba",
            huggingface_id="Qwen/Qwen2.5-72B-Instruct",
            description="Alibaba's advanced multilingual model",
            capabilities=["text_generation", "reasoning", "math", "code_generation"],
            license="Apache 2.0"
        ))
        
        self.register_model(ModelInfo(
            name="DeepSeek-V3",
            model_type=ModelType.OPEN_SOURCE,
            category=ModelCategory.LANGUAGE_MODEL,
            parameters=671_000_000_000,
            context_length=64000,
            provider="DeepSeek",
            huggingface_id="deepseek-ai/deepseek-llm-67b-chat",
            description="DeepSeek's flagship model with MoE architecture",
            capabilities=["text_generation", "reasoning", "math", "code_generation"],
            license="DeepSeek License"
        ))
        
        self.register_model(ModelInfo(
            name="Mixtral-8x22B",
            model_type=ModelType.OPEN_SOURCE,
            category=ModelCategory.LANGUAGE_MODEL,
            parameters=141_000_000_000,
            context_length=65536,
            provider="Mistral AI",
            huggingface_id="mistralai/Mixtral-8x22B-Instruct-v0.1",
            description="Mistral's sparse mixture of experts model",
            capabilities=["text_generation", "reasoning", "multilingual"],
            license="Apache 2.0"
        ))
        
        self.register_model(ModelInfo(
            name="Claude-3.5-Sonnet",
            model_type=ModelType.CLOSED_SOURCE,
            category=ModelCategory.LANGUAGE_MODEL,
            parameters=None,  # Unknown
            context_length=200000,
            provider="Anthropic",
            api_endpoint="https://api.anthropic.com/v1/messages",
            description="Anthropic's most capable model",
            capabilities=["text_generation", "reasoning", "code_generation", "analysis"],
            license="Proprietary"
        ))
        
        self.register_model(ModelInfo(
            name="GPT-4o",
            model_type=ModelType.CLOSED_SOURCE,
            category=ModelCategory.MULTIMODAL,
            parameters=None,  # Unknown
            context_length=128000,
            provider="OpenAI",
            api_endpoint="https://api.openai.com/v1/chat/completions",
            description="OpenAI's flagship multimodal model",
            capabilities=["text_generation", "vision", "reasoning", "code_generation"],
            license="Proprietary"
        ))
        
        self.register_model(ModelInfo(
            name="Gemini-1.5-Pro",
            model_type=ModelType.CLOSED_SOURCE,
            category=ModelCategory.MULTIMODAL,
            parameters=None,  # Unknown
            context_length=2000000,
            provider="Google",
            api_endpoint="https://generativelanguage.googleapis.com/v1beta/models",
            description="Google's advanced multimodal model with massive context",
            capabilities=["text_generation", "vision", "reasoning", "long_context"],
            license="Proprietary"
        ))
        
        self.register_model(ModelInfo(
            name="CodeLlama-70B",
            model_type=ModelType.OPEN_SOURCE,
            category=ModelCategory.CODE_GENERATION,
            parameters=70_000_000_000,
            context_length=100000,
            provider="Meta",
            huggingface_id="codellama/CodeLlama-70b-Instruct-hf",
            description="Meta's specialized code generation model",
            capabilities=["code_generation", "code_completion", "debugging"],
            license="Llama 2 Community License"
        ))
        
        self.register_model(ModelInfo(
            name="Qwen2.5-Math-72B",
            model_type=ModelType.OPEN_SOURCE,
            category=ModelCategory.MATH,
            parameters=72_000_000_000,
            context_length=32768,
            provider="Alibaba",
            huggingface_id="Qwen/Qwen2.5-Math-72B-Instruct",
            description="Specialized mathematical reasoning model",
            capabilities=["math", "reasoning", "problem_solving"],
            license="Apache 2.0"
        ))
        
        self.register_model(ModelInfo(
            name="LLaVA-1.6-34B",
            model_type=ModelType.OPEN_SOURCE,
            category=ModelCategory.VISION,
            parameters=34_000_000_000,
            context_length=4096,
            provider="Microsoft",
            huggingface_id="liuhaotian/llava-v1.6-34b",
            description="Large Language and Vision Assistant",
            capabilities=["vision", "multimodal", "image_understanding"],
            license="Apache 2.0"
        ))
    
    def register_model(self, model_info: ModelInfo):
        """Register a new model in the registry."""
        self.models[model_info.name] = model_info
    
    def get_model(self, name: str) -> Optional[ModelInfo]:
        """Get model information by name."""
        return self.models.get(name)
    
    def get_models_by_type(self, model_type: ModelType) -> List[ModelInfo]:
        """Get all models of a specific type."""
        return [model for model in self.models.values() if model.model_type == model_type]
    
    def get_models_by_category(self, category: ModelCategory) -> List[ModelInfo]:
        """Get all models in a specific category."""
        return [model for model in self.models.values() if model.category == category]
    
    def get_top_models_by_parameters(self, limit: int = 10) -> List[ModelInfo]:
        """Get top models by parameter count."""
        models_with_params = [m for m in self.models.values() if m.parameters is not None]
        return sorted(models_with_params, key=lambda x: x.parameters, reverse=True)[:limit]
    
    def get_benchmark_comparison_set(self) -> Dict[str, List[ModelInfo]]:
        """Get a comprehensive set of models for benchmarking comparison."""
        return {
            "truthgpt_models": self.get_models_by_type(ModelType.TRUTHGPT),
            "open_source_leaders": [
                self.get_model("Llama-3.1-405B"),
                self.get_model("Qwen2.5-72B"),
                self.get_model("DeepSeek-V3"),
                self.get_model("Mixtral-8x22B"),
                self.get_model("CodeLlama-70B"),
                self.get_model("Qwen2.5-Math-72B")
            ],
            "closed_source_leaders": [
                self.get_model("Claude-3.5-Sonnet"),
                self.get_model("GPT-4o"),
                self.get_model("Gemini-1.5-Pro")
            ],
            "multimodal_models": [
                self.get_model("TruthGPT-Viral-Clipper"),
                self.get_model("TruthGPT-Brand-Analyzer"),
                self.get_model("GPT-4o"),
                self.get_model("Gemini-1.5-Pro"),
                self.get_model("LLaVA-1.6-34B")
            ]
        }
    
    def get_best_models_only(self) -> Dict[str, List[ModelInfo]]:
        """Get only the absolute best performing models for streamlined comparison."""
        return {
            "truthgpt_models": self.get_models_by_type(ModelType.TRUTHGPT),
            "open_source_best": [
                self.get_model("Llama-3.1-405B"),
                self.get_model("Qwen2.5-72B"),
                self.get_model("DeepSeek-V3")
            ],
            "closed_source_best": [
                self.get_model("Claude-3.5-Sonnet"),
                self.get_model("GPT-4o"),
                self.get_model("Gemini-1.5-Pro")
            ]
        }
    
    def export_registry(self, filepath: str):
        """Export the model registry to JSON."""
        registry_data = {}
        for name, model in self.models.items():
            registry_data[name] = {
                "name": model.name,
                "model_type": model.model_type.value,
                "category": model.category.value,
                "parameters": model.parameters,
                "context_length": model.context_length,
                "provider": model.provider,
                "api_endpoint": model.api_endpoint,
                "local_path": model.local_path,
                "huggingface_id": model.huggingface_id,
                "description": model.description,
                "capabilities": model.capabilities,
                "license": model.license
            }
        
        with open(filepath, 'w') as f:
            json.dump(registry_data, f, indent=2)
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get a summary of all registered models."""
        summary = {
            "total_models": len(self.models),
            "by_type": {},
            "by_category": {},
            "by_provider": {}
        }
        
        for model in self.models.values():
            type_key = model.model_type.value
            summary["by_type"][type_key] = summary["by_type"].get(type_key, 0) + 1
            
            cat_key = model.category.value
            summary["by_category"][cat_key] = summary["by_category"].get(cat_key, 0) + 1
            
            summary["by_provider"][model.provider] = summary["by_provider"].get(model.provider, 0) + 1
        
        return summary

if __name__ == "__main__":
    registry = ModelRegistry()
    
    print("🚀 TruthGPT Model Registry Initialized")
    print("=" * 50)
    
    summary = registry.get_model_summary()
    print(f"Total Models: {summary['total_models']}")
    print(f"By Type: {summary['by_type']}")
    print(f"By Category: {summary['by_category']}")
    print(f"By Provider: {summary['by_provider']}")
    
    print("\n📊 Benchmark Comparison Set:")
    comparison_set = registry.get_benchmark_comparison_set()
    for category, models in comparison_set.items():
        print(f"\n{category.upper()}:")
        for model in models:
            if model:
                params = f"{model.parameters:,}" if model.parameters else "Unknown"
                print(f"  - {model.name} ({params} params) - {model.provider}")
    
    registry.export_registry("/home/ubuntu/TruthGPT/benchmarking_framework/model_registry.json")
    print("\n✅ Model registry exported to model_registry.json")
