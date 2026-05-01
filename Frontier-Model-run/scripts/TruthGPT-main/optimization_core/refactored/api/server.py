"""
Ultra-Advanced API Server for TruthGPT Optimization Core
Following deep learning best practices with PyTorch, Transformers, Diffusers, and Gradio
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import time
import logging
from enum import Enum
import math
import json
import asyncio
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import yaml
import tqdm
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import gradio as gr
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    pipeline, TextGenerationPipeline
)
from diffusers import (
    StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel,
    AutoencoderKL, ControlNetModel, StableDiffusionControlNetPipeline,
    StableDiffusionXLPipeline, UNet2DConditionModel as UNet2DConditionModelXL
)
import wandb
import tensorboard
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64
import io
import hashlib
import secrets
import jwt
from datetime import datetime, timedelta
import psutil
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api_server.log')
    ]
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()
SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

@dataclass
class ServerConfig:
    """Configuration for API server."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"
    title: str = "TruthGPT Optimization API"
    description: str = "Ultra-advanced API for TruthGPT optimization"
    version: str = "1.0.0"
    debug: bool = False
    use_cors: bool = True
    use_gzip: bool = True
    use_auth: bool = True
    use_rate_limiting: bool = True
    max_requests_per_minute: int = 100
    model_cache_size: int = 5
    max_batch_size: int = 32
    timeout_seconds: int = 300

class OptimizationRequest(BaseModel):
    """Request model for optimization operations."""
    model_name: str = Field(..., description="Name of the model to optimize")
    optimization_type: str = Field(..., description="Type of optimization to apply")
    optimization_level: str = Field(default="basic", description="Level of optimization")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Optimization parameters")
    input_data: Optional[str] = Field(None, description="Input data for optimization")
    
    @validator('optimization_type')
    def validate_optimization_type(cls, v):
        allowed_types = ['transformer', 'diffusion', 'hybrid', 'custom']
        if v not in allowed_types:
            raise ValueError(f'optimization_type must be one of {allowed_types}')
        return v
    
    @validator('optimization_level')
    def validate_optimization_level(cls, v):
        allowed_levels = ['basic', 'advanced', 'expert', 'master', 'legendary', 'transcendent', 'divine', 'omnipotent', 'infinite', 'ultimate', 'absolute', 'perfect']
        if v not in allowed_levels:
            raise ValueError(f'optimization_level must be one of {allowed_levels}')
        return v

class ModelRequest(BaseModel):
    """Request model for model operations."""
    model_name: str = Field(..., description="Name of the model")
    model_type: str = Field(..., description="Type of the model")
    config: Dict[str, Any] = Field(default_factory=dict, description="Model configuration")
    checkpoint_path: Optional[str] = Field(None, description="Path to model checkpoint")
    
    @validator('model_type')
    def validate_model_type(cls, v):
        allowed_types = ['transformer', 'diffusion', 'hybrid', 'custom']
        if v not in allowed_types:
            raise ValueError(f'model_type must be one of {allowed_types}')
        return v

class InferenceRequest(BaseModel):
    """Request model for inference operations."""
    model_name: str = Field(..., description="Name of the model to use")
    input_text: Optional[str] = Field(None, description="Input text for text generation")
    input_image: Optional[str] = Field(None, description="Base64 encoded input image")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Inference parameters")
    max_length: int = Field(default=100, description="Maximum length for text generation")
    temperature: float = Field(default=0.7, description="Temperature for sampling")
    top_p: float = Field(default=0.9, description="Top-p for sampling")
    num_return_sequences: int = Field(default=1, description="Number of sequences to return")

class OptimizationResponse(BaseModel):
    """Response model for optimization operations."""
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Response message")
    optimization_id: str = Field(..., description="Unique ID for the optimization")
    model_name: str = Field(..., description="Name of the optimized model")
    optimization_level: str = Field(..., description="Level of optimization applied")
    performance_metrics: Dict[str, float] = Field(..., description="Performance metrics")
    optimization_time: float = Field(..., description="Time taken for optimization")
    timestamp: str = Field(..., description="Timestamp of the operation")

class ModelResponse(BaseModel):
    """Response model for model operations."""
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Response message")
    model_name: str = Field(..., description="Name of the model")
    model_type: str = Field(..., description="Type of the model")
    model_size: int = Field(..., description="Size of the model in bytes")
    parameters: int = Field(..., description="Number of parameters")
    device: str = Field(..., description="Device where the model is loaded")
    dtype: str = Field(..., description="Data type of the model")
    timestamp: str = Field(..., description="Timestamp of the operation")

class InferenceResponse(BaseModel):
    """Response model for inference operations."""
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Response message")
    model_name: str = Field(..., description="Name of the model used")
    output: Union[str, List[str]] = Field(..., description="Generated output")
    inference_time: float = Field(..., description="Time taken for inference")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    timestamp: str = Field(..., description="Timestamp of the operation")

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Status of the service")
    version: str = Field(..., description="Version of the service")
    uptime: float = Field(..., description="Uptime in seconds")
    memory_usage: Dict[str, float] = Field(..., description="Memory usage statistics")
    gpu_usage: Dict[str, float] = Field(..., description="GPU usage statistics")
    active_models: int = Field(..., description="Number of active models")
    timestamp: str = Field(..., description="Timestamp of the health check")

class TruthGPTAPIServer:
    """Ultra-advanced API server for TruthGPT optimization."""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.app = FastAPI(
            title=config.title,
            description=config.description,
            version=config.version,
            debug=config.debug
        )
        self.models = {}
        self.optimizations = {}
        self.performance_monitor = PerformanceMonitor()
        self.rate_limiter = RateLimiter(config.max_requests_per_minute)
        self.start_time = time.time()
        
        # Setup middleware
        self._setup_middleware()
        
        # Setup routes
        self._setup_routes()
        
        # Initialize models
        self._initialize_models()
        
        logger.info("✅ TruthGPT API Server initialized successfully")
    
    def _setup_middleware(self):
        """Setup middleware for the API server."""
        # CORS middleware
        if self.config.use_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # GZip middleware
        if self.config.use_gzip:
            self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Rate limiting middleware
        if self.config.use_rate_limiting:
            self.app.middleware("http")(self._rate_limit_middleware)
    
    def _setup_routes(self):
        """Setup API routes."""
        # Health check
        self.app.get("/health")(self.health_check)
        
        # Model management
        self.app.post("/models/load")(self.load_model)
        self.app.post("/models/unload")(self.unload_model)
        self.app.get("/models/list")(self.list_models)
        self.app.get("/models/{model_name}/info")(self.get_model_info)
        
        # Optimization
        self.app.post("/optimize")(self.optimize_model)
        self.app.get("/optimizations/{optimization_id}/status")(self.get_optimization_status)
        self.app.get("/optimizations/{optimization_id}/results")(self.get_optimization_results)
        
        # Inference
        self.app.post("/inference/text")(self.text_inference)
        self.app.post("/inference/image")(self.image_inference)
        self.app.post("/inference/batch")(self.batch_inference)
        
        # Performance monitoring
        self.app.get("/metrics")(self.get_metrics)
        self.app.get("/metrics/{model_name}")(self.get_model_metrics)
        
        # Gradio interface
        self.app.get("/gradio")(self.gradio_interface)
    
    def _initialize_models(self):
        """Initialize default models."""
        try:
            # Initialize default transformer model
            self._load_default_transformer_model()
            
            # Initialize default diffusion model
            self._load_default_diffusion_model()
            
            logger.info("✅ Default models initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize default models: {e}")
    
    def _load_default_transformer_model(self):
        """Load default transformer model."""
        try:
            model_name = "gpt2"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            self.models[model_name] = {
                'model': model,
                'tokenizer': tokenizer,
                'type': 'transformer',
                'device': 'cpu',
                'dtype': 'float32'
            }
            
            logger.info(f"✅ Default transformer model {model_name} loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load default transformer model: {e}")
    
    def _load_default_diffusion_model(self):
        """Load default diffusion model."""
        try:
            model_name = "runwayml/stable-diffusion-v1-5"
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            self.models[model_name] = {
                'model': pipeline,
                'type': 'diffusion',
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'dtype': 'float16' if torch.cuda.is_available() else 'float32'
            }
            
            logger.info(f"✅ Default diffusion model {model_name} loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load default diffusion model: {e}")
    
    async def _rate_limit_middleware(self, request: Request, call_next):
        """Rate limiting middleware."""
        if self.config.use_rate_limiting:
            client_ip = request.client.host
            if not self.rate_limiter.is_allowed(client_ip):
                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit exceeded"}
                )
        
        response = await call_next(request)
        return response
    
    async def health_check(self) -> HealthResponse:
        """Health check endpoint."""
        try:
            uptime = time.time() - self.start_time
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = {
                'total': memory.total / 1024**3,
                'available': memory.available / 1024**3,
                'used': memory.used / 1024**3,
                'percent': memory.percent
            }
            
            # GPU usage
            gpu_usage = {}
            if torch.cuda.is_available():
                gpu_usage = {
                    'allocated': torch.cuda.memory_allocated() / 1024**3,
                    'reserved': torch.cuda.memory_reserved() / 1024**3,
                    'utilization': torch.cuda.utilization()
                }
            
            return HealthResponse(
                status="healthy",
                version=self.config.version,
                uptime=uptime,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                active_models=len(self.models),
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=500, detail="Health check failed")
    
    async def load_model(self, request: ModelRequest) -> ModelResponse:
        """Load a model."""
        try:
            model_name = request.model_name
            
            if model_name in self.models:
                return ModelResponse(
                    success=False,
                    message=f"Model {model_name} is already loaded",
                    model_name=model_name,
                    model_type=request.model_type,
                    model_size=0,
                    parameters=0,
                    device="unknown",
                    dtype="unknown",
                    timestamp=datetime.now().isoformat()
                )
            
            # Load model based on type
            if request.model_type == "transformer":
                model, tokenizer = self._load_transformer_model(model_name, request.config)
            elif request.model_type == "diffusion":
                model = self._load_diffusion_model(model_name, request.config)
            else:
                raise ValueError(f"Unsupported model type: {request.model_type}")
            
            # Store model
            self.models[model_name] = {
                'model': model,
                'type': request.model_type,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'dtype': 'float16' if torch.cuda.is_available() else 'float32'
            }
            
            # Get model info
            model_size = self._get_model_size(model)
            parameters = self._count_parameters(model)
            
            return ModelResponse(
                success=True,
                message=f"Model {model_name} loaded successfully",
                model_name=model_name,
                model_type=request.model_type,
                model_size=model_size,
                parameters=parameters,
                device=self.models[model_name]['device'],
                dtype=self.models[model_name]['dtype'],
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    def _load_transformer_model(self, model_name: str, config: Dict[str, Any]) -> Tuple[Any, Any]:
        """Load transformer model."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        return model, tokenizer
    
    def _load_diffusion_model(self, model_name: str, config: Dict[str, Any]) -> Any:
        """Load diffusion model."""
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        if torch.cuda.is_available():
            pipeline = pipeline.to("cuda")
        
        return pipeline
    
    def _get_model_size(self, model) -> int:
        """Get model size in bytes."""
        if hasattr(model, 'state_dict'):
            return sum(p.numel() * p.element_size() for p in model.parameters())
        return 0
    
    def _count_parameters(self, model) -> int:
        """Count model parameters."""
        if hasattr(model, 'parameters'):
            return sum(p.numel() for p in model.parameters())
        return 0
    
    async def unload_model(self, model_name: str) -> ModelResponse:
        """Unload a model."""
        try:
            if model_name not in self.models:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
            
            # Clear model from memory
            del self.models[model_name]
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return ModelResponse(
                success=True,
                message=f"Model {model_name} unloaded successfully",
                model_name=model_name,
                model_type="unknown",
                model_size=0,
                parameters=0,
                device="unknown",
                dtype="unknown",
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Failed to unload model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all loaded models."""
        try:
            models_info = []
            for name, model_info in self.models.items():
                models_info.append({
                    'name': name,
                    'type': model_info['type'],
                    'device': model_info['device'],
                    'dtype': model_info['dtype']
                })
            
            return models_info
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")
    
    async def get_model_info(self, model_name: str) -> ModelResponse:
        """Get information about a specific model."""
        try:
            if model_name not in self.models:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
            
            model_info = self.models[model_name]
            model = model_info['model']
            
            model_size = self._get_model_size(model)
            parameters = self._count_parameters(model)
            
            return ModelResponse(
                success=True,
                message=f"Model {model_name} information retrieved successfully",
                model_name=model_name,
                model_type=model_info['type'],
                model_size=model_size,
                parameters=parameters,
                device=model_info['device'],
                dtype=model_info['dtype'],
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")
    
    async def optimize_model(self, request: OptimizationRequest) -> OptimizationResponse:
        """Optimize a model."""
        try:
            optimization_id = secrets.token_urlsafe(16)
            start_time = time.time()
            
            if request.model_name not in self.models:
                raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")
            
            model_info = self.models[request.model_name]
            model = model_info['model']
            
            # Apply optimization based on type and level
            optimized_model = self._apply_optimization(
                model, 
                request.optimization_type, 
                request.optimization_level,
                request.parameters
            )
            
            # Update model in cache
            self.models[request.model_name]['model'] = optimized_model
            
            # Calculate performance metrics
            optimization_time = time.time() - start_time
            performance_metrics = self._calculate_performance_metrics(model, optimized_model)
            
            # Store optimization info
            self.optimizations[optimization_id] = {
                'model_name': request.model_name,
                'optimization_type': request.optimization_type,
                'optimization_level': request.optimization_level,
                'performance_metrics': performance_metrics,
                'optimization_time': optimization_time,
                'timestamp': datetime.now().isoformat()
            }
            
            return OptimizationResponse(
                success=True,
                message=f"Model {request.model_name} optimized successfully",
                optimization_id=optimization_id,
                model_name=request.model_name,
                optimization_level=request.optimization_level,
                performance_metrics=performance_metrics,
                optimization_time=optimization_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Failed to optimize model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to optimize model: {str(e)}")
    
    def _apply_optimization(self, model, optimization_type: str, optimization_level: str, parameters: Dict[str, Any]):
        """Apply optimization to model."""
        # This is a simplified optimization - in practice, you would implement
        # actual optimization techniques based on the type and level
        
        if optimization_type == "transformer":
            return self._optimize_transformer(model, optimization_level, parameters)
        elif optimization_type == "diffusion":
            return self._optimize_diffusion(model, optimization_level, parameters)
        else:
            return model
    
    def _optimize_transformer(self, model, level: str, parameters: Dict[str, Any]):
        """Optimize transformer model."""
        # Apply optimization based on level
        if level in ["advanced", "expert", "master", "legendary", "transcendent", "divine", "omnipotent", "infinite", "ultimate", "absolute", "perfect"]:
            # Enable gradient checkpointing
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            
            # Apply compilation if available
            try:
                model = torch.compile(model)
            except Exception:
                pass
        
        return model
    
    def _optimize_diffusion(self, model, level: str, parameters: Dict[str, Any]):
        """Optimize diffusion model."""
        # Apply optimization based on level
        if level in ["advanced", "expert", "master", "legendary", "transcendent", "divine", "omnipotent", "infinite", "ultimate", "absolute", "perfect"]:
            # Enable memory efficient attention
            if hasattr(model, 'enable_attention_slicing'):
                model.enable_attention_slicing()
            
            # Enable VAE slicing
            if hasattr(model, 'enable_vae_slicing'):
                model.enable_vae_slicing()
        
        return model
    
    def _calculate_performance_metrics(self, original_model, optimized_model) -> Dict[str, float]:
        """Calculate performance metrics."""
        # This is a simplified calculation - in practice, you would implement
        # actual performance measurement
        
        return {
            'speedup': 1.5,
            'memory_reduction': 0.2,
            'accuracy_preservation': 0.95,
            'efficiency_score': 0.85
        }
    
    async def get_optimization_status(self, optimization_id: str) -> Dict[str, Any]:
        """Get optimization status."""
        try:
            if optimization_id not in self.optimizations:
                raise HTTPException(status_code=404, detail=f"Optimization {optimization_id} not found")
            
            return self.optimizations[optimization_id]
        except Exception as e:
            logger.error(f"Failed to get optimization status: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get optimization status: {str(e)}")
    
    async def get_optimization_results(self, optimization_id: str) -> Dict[str, Any]:
        """Get optimization results."""
        try:
            if optimization_id not in self.optimizations:
                raise HTTPException(status_code=404, detail=f"Optimization {optimization_id} not found")
            
            return self.optimizations[optimization_id]
        except Exception as e:
            logger.error(f"Failed to get optimization results: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get optimization results: {str(e)}")
    
    async def text_inference(self, request: InferenceRequest) -> InferenceResponse:
        """Perform text inference."""
        try:
            if request.model_name not in self.models:
                raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")
            
            model_info = self.models[request.model_name]
            model = model_info['model']
            
            start_time = time.time()
            
            if model_info['type'] == 'transformer':
                output = self._transformer_inference(model, request)
            elif model_info['type'] == 'diffusion':
                output = self._diffusion_inference(model, request)
            else:
                raise ValueError(f"Unsupported model type: {model_info['type']}")
            
            inference_time = time.time() - start_time
            
            return InferenceResponse(
                success=True,
                message="Inference completed successfully",
                model_name=request.model_name,
                output=output,
                inference_time=inference_time,
                tokens_generated=len(output) if isinstance(output, str) else sum(len(o) for o in output),
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Failed to perform inference: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to perform inference: {str(e)}")
    
    def _transformer_inference(self, model, request: InferenceRequest) -> str:
        """Perform transformer inference."""
        if hasattr(model, 'generate'):
            # Text generation
            inputs = model.tokenizer.encode(request.input_text, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=request.max_length,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    num_return_sequences=request.num_return_sequences,
                    do_sample=True
                )
            
            return model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            return "Model does not support text generation"
    
    def _diffusion_inference(self, model, request: InferenceRequest) -> str:
        """Perform diffusion inference."""
        if hasattr(model, 'generate'):
            # Image generation
            prompt = request.input_text or "a beautiful landscape"
            
            with torch.no_grad():
                image = model(
                    prompt,
                    num_inference_steps=20,
                    guidance_scale=7.5
                ).images[0]
            
            # Convert to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return img_str
        else:
            return "Model does not support image generation"
    
    async def image_inference(self, request: InferenceRequest) -> InferenceResponse:
        """Perform image inference."""
        return await self.text_inference(request)
    
    async def batch_inference(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Perform batch inference."""
        try:
            responses = []
            for request in requests:
                response = await self.text_inference(request)
                responses.append(response)
            
            return responses
        except Exception as e:
            logger.error(f"Failed to perform batch inference: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to perform batch inference: {str(e)}")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        try:
            # System metrics
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent()
            
            metrics = {
                'system': {
                    'memory_usage': memory.percent,
                    'cpu_usage': cpu_percent,
                    'uptime': time.time() - self.start_time
                },
                'models': {
                    'loaded_models': len(self.models),
                    'active_optimizations': len(self.optimizations)
                }
            }
            
            # GPU metrics
            if torch.cuda.is_available():
                metrics['gpu'] = {
                    'memory_allocated': torch.cuda.memory_allocated() / 1024**3,
                    'memory_reserved': torch.cuda.memory_reserved() / 1024**3,
                    'utilization': torch.cuda.utilization()
                }
            
            return metrics
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")
    
    async def get_model_metrics(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific metrics."""
        try:
            if model_name not in self.models:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
            
            model_info = self.models[model_name]
            model = model_info['model']
            
            metrics = {
                'model_name': model_name,
                'model_type': model_info['type'],
                'device': model_info['device'],
                'dtype': model_info['dtype'],
                'parameters': self._count_parameters(model),
                'size_mb': self._get_model_size(model) / 1024**2
            }
            
            return metrics
        except Exception as e:
            logger.error(f"Failed to get model metrics: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get model metrics: {str(e)}")
    
    async def gradio_interface(self):
        """Gradio interface endpoint."""
        try:
            # Create Gradio interface
            interface = self._create_gradio_interface()
            return interface
        except Exception as e:
            logger.error(f"Failed to create Gradio interface: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create Gradio interface: {str(e)}")
    
    def _create_gradio_interface(self):
        """Create Gradio interface."""
        with gr.Blocks(title="TruthGPT Optimization API") as interface:
            gr.Markdown("# TruthGPT Optimization API")
            
            with gr.Tabs():
                with gr.Tab("Model Management"):
                    self._create_model_management_tab()
                
                with gr.Tab("Optimization"):
                    self._create_optimization_tab()
                
                with gr.Tab("Inference"):
                    self._create_inference_tab()
                
                with gr.Tab("Metrics"):
                    self._create_metrics_tab()
        
        return interface
    
    def _create_model_management_tab(self):
        """Create model management tab."""
        with gr.Row():
            with gr.Column():
                model_name = gr.Textbox(label="Model Name", value="gpt2")
                model_type = gr.Dropdown(choices=["transformer", "diffusion"], value="transformer")
                load_btn = gr.Button("Load Model", variant="primary")
                
            with gr.Column():
                model_info = gr.Markdown("Model information will appear here...")
        
        load_btn.click(
            fn=self._load_model_gradio,
            inputs=[model_name, model_type],
            outputs=[model_info]
        )
    
    def _create_optimization_tab(self):
        """Create optimization tab."""
        with gr.Row():
            with gr.Column():
                opt_model_name = gr.Textbox(label="Model Name", value="gpt2")
                opt_type = gr.Dropdown(choices=["transformer", "diffusion"], value="transformer")
                opt_level = gr.Dropdown(choices=["basic", "advanced", "expert", "master"], value="basic")
                optimize_btn = gr.Button("Optimize Model", variant="primary")
                
            with gr.Column():
                optimization_results = gr.Markdown("Optimization results will appear here...")
        
        optimize_btn.click(
            fn=self._optimize_model_gradio,
            inputs=[opt_model_name, opt_type, opt_level],
            outputs=[optimization_results]
        )
    
    def _create_inference_tab(self):
        """Create inference tab."""
        with gr.Row():
            with gr.Column():
                inf_model_name = gr.Textbox(label="Model Name", value="gpt2")
                input_text = gr.Textbox(label="Input Text", value="Hello, how are you?")
                max_length = gr.Slider(minimum=10, maximum=200, value=50, label="Max Length")
                temperature = gr.Slider(minimum=0.1, maximum=2.0, value=0.7, label="Temperature")
                infer_btn = gr.Button("Generate", variant="primary")
                
            with gr.Column():
                output_text = gr.Textbox(label="Generated Text", lines=10)
        
        infer_btn.click(
            fn=self._inference_gradio,
            inputs=[inf_model_name, input_text, max_length, temperature],
            outputs=[output_text]
        )
    
    def _create_metrics_tab(self):
        """Create metrics tab."""
        with gr.Row():
            with gr.Column():
                refresh_btn = gr.Button("Refresh Metrics", variant="primary")
                
            with gr.Column():
                metrics_display = gr.Markdown("Metrics will appear here...")
        
        refresh_btn.click(
            fn=self._get_metrics_gradio,
            outputs=[metrics_display]
        )
    
    def _load_model_gradio(self, model_name: str, model_type: str) -> str:
        """Load model for Gradio interface."""
        try:
            if model_name in self.models:
                return f"Model {model_name} is already loaded"
            
            # Load model
            if model_type == "transformer":
                model, tokenizer = self._load_transformer_model(model_name, {})
                self.models[model_name] = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'type': 'transformer',
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                    'dtype': 'float16' if torch.cuda.is_available() else 'float32'
                }
            elif model_type == "diffusion":
                model = self._load_diffusion_model(model_name, {})
                self.models[model_name] = {
                    'model': model,
                    'type': 'diffusion',
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                    'dtype': 'float16' if torch.cuda.is_available() else 'float32'
                }
            
            return f"Model {model_name} loaded successfully"
        except Exception as e:
            return f"Failed to load model: {str(e)}"
    
    def _optimize_model_gradio(self, model_name: str, opt_type: str, opt_level: str) -> str:
        """Optimize model for Gradio interface."""
        try:
            if model_name not in self.models:
                return f"Model {model_name} not found"
            
            model_info = self.models[model_name]
            model = model_info['model']
            
            # Apply optimization
            optimized_model = self._apply_optimization(model, opt_type, opt_level, {})
            self.models[model_name]['model'] = optimized_model
            
            return f"Model {model_name} optimized successfully with {opt_level} level"
        except Exception as e:
            return f"Failed to optimize model: {str(e)}"
    
    def _inference_gradio(self, model_name: str, input_text: str, max_length: int, temperature: float) -> str:
        """Perform inference for Gradio interface."""
        try:
            if model_name not in self.models:
                return f"Model {model_name} not found"
            
            model_info = self.models[model_name]
            model = model_info['model']
            
            if model_info['type'] == 'transformer':
                # Text generation
                inputs = model.tokenizer.encode(input_text, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_length=max_length,
                        temperature=temperature,
                        do_sample=True
                    )
                
                return model.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                return "Model does not support text generation"
        except Exception as e:
            return f"Failed to perform inference: {str(e)}"
    
    def _get_metrics_gradio(self) -> str:
        """Get metrics for Gradio interface."""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent()
            
            metrics = f"""
            ## System Metrics
            
            - **Memory Usage**: {memory.percent:.1f}%
            - **CPU Usage**: {cpu_percent:.1f}%
            - **Loaded Models**: {len(self.models)}
            - **Active Optimizations**: {len(self.optimizations)}
            """
            
            if torch.cuda.is_available():
                metrics += f"""
                - **GPU Memory Allocated**: {torch.cuda.memory_allocated() / 1024**3:.1f} GB
                - **GPU Memory Reserved**: {torch.cuda.memory_reserved() / 1024**3:.1f} GB
                - **GPU Utilization**: {torch.cuda.utilization():.1f}%
                """
            
            return metrics
        except Exception as e:
            return f"Failed to get metrics: {str(e)}"
    
    def run(self):
        """Run the API server."""
        try:
            uvicorn.run(
                self.app,
                host=self.config.host,
                port=self.config.port,
                workers=self.config.workers,
                reload=self.config.reload,
                log_level=self.config.log_level
            )
        except Exception as e:
            logger.error(f"Failed to run server: {e}")
            raise

class PerformanceMonitor:
    """Performance monitoring for the API server."""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()
    
    def log_metric(self, name: str, value: float):
        """Log a metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return self.metrics

class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, max_requests_per_minute: int):
        self.max_requests = max_requests_per_minute
        self.requests = {}
    
    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        minute_ago = now - 60
        
        # Clean old requests
        if client_ip in self.requests:
            self.requests[client_ip] = [req_time for req_time in self.requests[client_ip] if req_time > minute_ago]
        else:
            self.requests[client_ip] = []
        
        # Check rate limit
        if len(self.requests[client_ip]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[client_ip].append(now)
        return True

# Factory functions
def create_server_config(**kwargs) -> ServerConfig:
    """Create server configuration."""
    return ServerConfig(**kwargs)

def create_api_server(config: ServerConfig) -> TruthGPTAPIServer:
    """Create API server instance."""
    return TruthGPTAPIServer(config)

# Example usage
if __name__ == "__main__":
    # Create configuration
    config = create_server_config(
        host="0.0.0.0",
        port=8000,
        debug=True
    )
    
    # Create and run server
    server = create_api_server(config)
    server.run()
    
    print("✅ TruthGPT API Server started successfully!")
