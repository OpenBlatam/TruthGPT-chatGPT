"""
Diffusion Models management using Diffusers library.
Supports Stable Diffusion, SDXL, and custom diffusion models.
"""
import logging
from typing import Optional, Dict, Any, Union, List
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    UNet2DConditionModel,
)
from diffusers.models.attention_processor import AttnProcessor2_0

logger = logging.getLogger(__name__)


class DiffusionModelManager:
    """
    Manages diffusion models with support for different pipelines and schedulers.
    """
    
    def __init__(self):
        """Initialize diffusion model manager."""
        self.pipeline: Optional[Any] = None
        self.device: Optional[torch.device] = None
    
    def load_pipeline(
        self,
        model_id: str,
        pipeline_type: str = "stable-diffusion",
        variant: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16,
        device: Optional[Union[str, torch.device]] = None,
        scheduler_type: Optional[str] = None,
        enable_attention_slicing: bool = True,
        enable_vae_slicing: bool = True,
        enable_vae_tiling: bool = False,
    ) -> Any:
        """
        Load diffusion pipeline.
        
        Args:
            model_id: Model identifier (HuggingFace repo or local path)
            pipeline_type: Type of pipeline (stable-diffusion|stable-diffusion-xl)
            variant: Optional variant (fp16|bf16)
            torch_dtype: Data type for model weights
            device: Device to load on
            scheduler_type: Optional scheduler (ddim|dpm|euler)
            enable_attention_slicing: Enable attention slicing for memory
            enable_vae_slicing: Enable VAE slicing for memory
            enable_vae_tiling: Enable VAE tiling for memory
        
        Returns:
            Loaded pipeline
        """
        try:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            elif isinstance(device, str):
                device = torch.device(device)
            
            self.device = device
            logger.info(f"Loading {pipeline_type} pipeline: {model_id} on {device}")
            
            # Load appropriate pipeline
            if pipeline_type == "stable-diffusion-xl":
                pipeline_cls = StableDiffusionXLPipeline
            else:
                pipeline_cls = StableDiffusionPipeline
            
            # Load with optional variant
            load_kwargs = {
                "torch_dtype": torch_dtype,
                "variant": variant,
            } if variant else {"torch_dtype": torch_dtype}
            
            pipeline = pipeline_cls.from_pretrained(
                model_id,
                **load_kwargs
            )
            
            # Set scheduler if specified
            if scheduler_type:
                pipeline = self._set_scheduler(pipeline, scheduler_type)
            
            # Move to device
            pipeline = pipeline.to(device)
            
            # Enable memory optimizations
            if enable_attention_slicing:
                pipeline.enable_attention_slicing()
                logger.debug("Attention slicing enabled")
            
            if enable_vae_slicing:
                pipeline.enable_vae_slicing()
                logger.debug("VAE slicing enabled")
            
            if enable_vae_tiling:
                pipeline.enable_vae_tiling()
                logger.debug("VAE tiling enabled")
            
            self.pipeline = pipeline
            logger.info("Pipeline loaded successfully")
            return pipeline
            
        except Exception as e:
            logger.error(f"Error loading diffusion pipeline: {e}", exc_info=True)
            raise
    
    def _set_scheduler(
        self,
        pipeline: Any,
        scheduler_type: str
    ) -> Any:
        """
        Set scheduler for pipeline.
        
        Args:
            pipeline: Diffusion pipeline
            scheduler_type: Scheduler type
        
        Returns:
            Pipeline with updated scheduler
        """
        scheduler_map = {
            "ddim": DDIMScheduler,
            "dpm": DPMSolverMultistepScheduler,
            "euler": EulerAncestralDiscreteScheduler,
        }
        
        if scheduler_type not in scheduler_map:
            logger.warning(f"Unknown scheduler type: {scheduler_type}, using default")
            return pipeline
        
        try:
            scheduler_cls = scheduler_map[scheduler_type]
            pipeline.scheduler = scheduler_cls.from_config(pipeline.scheduler.config)
            logger.info(f"Scheduler set to {scheduler_type}")
            return pipeline
        except Exception as e:
            logger.warning(f"Failed to set scheduler: {e}, using default")
            return pipeline
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate images from text prompt.
        
        Args:
            prompt: Text prompt(s)
            negative_prompt: Optional negative prompt(s)
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            height: Image height (optional)
            width: Image width (optional)
            num_images_per_prompt: Number of images per prompt
            seed: Optional random seed
            **kwargs: Additional generation arguments
        
        Returns:
            Generated images tensor
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_pipeline() first.")
        
        try:
            # Set generator if seed provided
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Generate images
            with torch.cuda.amp.autocast():
                images = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=generator,
                    **kwargs
                ).images
            
            logger.debug(f"Generated {len(images)} image(s)")
            return images
            
        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            raise
    
    def enable_xformers(self) -> None:
        """Enable xFormers attention for faster inference."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded")
        
        try:
            self.pipeline.enable_xformers_memory_efficient_attention()
            logger.info("xFormers attention enabled")
        except Exception as e:
            logger.warning(f"Failed to enable xFormers: {e}")
    
    def enable_model_cpu_offload(self) -> None:
        """Enable CPU offloading for memory efficiency."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded")
        
        try:
            self.pipeline.enable_model_cpu_offload()
            logger.info("CPU offloading enabled")
        except Exception as e:
            logger.warning(f"Failed to enable CPU offloading: {e}")
    
    def enable_sequential_cpu_offload(self) -> None:
        """Enable sequential CPU offloading for maximum memory efficiency."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded")
        
        try:
            self.pipeline.enable_sequential_cpu_offload()
            logger.info("Sequential CPU offloading enabled")
        except Exception as e:
            logger.warning(f"Failed to enable sequential CPU offloading: {e}")


class DiffusionTrainer:
    """
    Trainer for fine-tuning diffusion models.
    """
    
    def __init__(
        self,
        pipeline: Any,
        learning_rate: float = 5e-6,
        use_8bit_adam: bool = True,
    ):
        """
        Initialize diffusion trainer.
        
        Args:
            pipeline: Diffusion pipeline
            learning_rate: Learning rate
            use_8bit_adam: Use 8-bit Adam optimizer
        """
        self.pipeline = pipeline
        self.learning_rate = learning_rate
        self.use_8bit_adam = use_8bit_adam
    
    def prepare_for_training(self) -> None:
        """Prepare pipeline components for training."""
        try:
            # Set UNet to training mode
            self.pipeline.unet.train()
            
            # Freeze VAE and text encoder
            self.pipeline.vae.requires_grad_(False)
            self.pipeline.text_encoder.requires_grad_(False)
            
            logger.info("Pipeline prepared for training")
            
        except Exception as e:
            logger.error(f"Error preparing for training: {e}", exc_info=True)
            raise
    
    def get_unet(self) -> UNet2DConditionModel:
        """Get UNet model for training."""
        return self.pipeline.unet



