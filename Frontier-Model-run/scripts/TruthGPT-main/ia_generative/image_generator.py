"""
Advanced image generation with diffusion models and brand consistency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import math

@dataclass
class DiffusionImageGeneratorArgs:
    """Configuration for diffusion-based image generator."""
    image_size: int = 512
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_channels: int = 3
    
    unet_channels: Optional[List[int]] = None
    unet_layers: Optional[List[int]] = None
    attention_resolutions: Optional[List[int]] = None
    
    enable_brand_conditioning: bool = True
    enable_layout_generation: bool = True
    enable_style_transfer: bool = True
    
    def __post_init__(self):
        if self.unet_channels is None:
            self.unet_channels = [320, 640, 1280, 1280]
        if self.unet_layers is None:
            self.unet_layers = [2, 2, 2, 2]
        if self.attention_resolutions is None:
            self.attention_resolutions = [4, 2, 1]

class TimeEmbedding(nn.Module):
    """Time embedding for diffusion process."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4)
        )
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        time_emb = self._get_timestep_embedding(time, self.dim)
        return self.time_mlp(time_emb)
    
    def _get_timestep_embedding(self, timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        
        return emb

class CrossAttentionBlock(nn.Module):
    """Cross-attention block for conditioning."""
    
    def __init__(self, dim: int, context_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(context_dim, dim, bias=False)
        self.to_v = nn.Linear(context_dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        
        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(context_dim)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        context = self.context_norm(context)
        
        batch_size, seq_len, _ = x.shape
        
        q = self.to_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        if hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(q, k, v)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_weights = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.to_out(attn_output)
        
        return residual + output

class ResNetBlock(nn.Module):
    """ResNet block for U-Net architecture."""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = self.residual_conv(x)
        
        h = self.block1(x)
        time_emb = self.time_mlp(time_emb)[:, :, None, None]
        h = h + time_emb
        h = self.block2(h)
        
        return h + residual

class UNetModel(nn.Module):
    """U-Net model for diffusion-based image generation."""
    
    def __init__(self, args: DiffusionImageGeneratorArgs):
        super().__init__()
        self.args = args
        
        time_emb_dim = args.unet_channels[0] * 4
        self.time_embedding = TimeEmbedding(args.unet_channels[0])
        
        self.input_conv = nn.Conv2d(args.num_channels, args.unet_channels[0], 3, padding=1)
        
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        
        in_ch = args.unet_channels[0]
        for i, (out_ch, num_layers) in enumerate(zip(args.unet_channels, args.unet_layers)):
            down_block = nn.ModuleList()
            
            for j in range(num_layers):
                down_block.append(ResNetBlock(in_ch, out_ch, time_emb_dim))
                in_ch = out_ch
                
                if args.image_size // (2 ** i) in args.attention_resolutions:
                    down_block.append(CrossAttentionBlock(out_ch, 768))
            
            if i < len(args.unet_channels) - 1:
                down_block.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            
            self.down_blocks.append(down_block)
        
        self.mid_block = nn.ModuleList([
            ResNetBlock(args.unet_channels[-1], args.unet_channels[-1], time_emb_dim),
            CrossAttentionBlock(args.unet_channels[-1], 768),
            ResNetBlock(args.unet_channels[-1], args.unet_channels[-1], time_emb_dim)
        ])
        
        for i, (out_ch, num_layers) in enumerate(zip(reversed(args.unet_channels), reversed(args.unet_layers))):
            up_block = nn.ModuleList()
            
            for j in range(num_layers + 1):
                up_block.append(ResNetBlock(in_ch + out_ch, out_ch, time_emb_dim))
                in_ch = out_ch
                
                if args.image_size // (2 ** (len(args.unet_channels) - 1 - i)) in args.attention_resolutions:
                    up_block.append(CrossAttentionBlock(out_ch, 768))
            
            if i < len(args.unet_channels) - 1:
                up_block.append(nn.ConvTranspose2d(out_ch, out_ch, 4, stride=2, padding=1))
            
            self.up_blocks.append(up_block)
        
        self.output_conv = nn.Sequential(
            nn.GroupNorm(32, args.unet_channels[0]),
            nn.SiLU(),
            nn.Conv2d(args.unet_channels[0], args.num_channels, 3, padding=1)
        )
        
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        time_emb = self.time_embedding(timesteps)
        
        h = self.input_conv(x)
        skip_connections = [h]
        
        for block in self.down_blocks:
            for layer in block:
                if isinstance(layer, ResNetBlock):
                    h = layer(h, time_emb)
                elif isinstance(layer, CrossAttentionBlock) and context is not None:
                    batch_size, channels, height, width = h.shape
                    h_flat = h.view(batch_size, channels, -1).transpose(1, 2)
                    h_flat = layer(h_flat, context)
                    h = h_flat.transpose(1, 2).view(batch_size, channels, height, width)
                else:
                    h = layer(h)
                
                if not isinstance(layer, nn.Conv2d) or layer.stride == (1, 1):
                    skip_connections.append(h)
        
        for layer in self.mid_block:
            if isinstance(layer, ResNetBlock):
                h = layer(h, time_emb)
            elif isinstance(layer, CrossAttentionBlock) and context is not None:
                batch_size, channels, height, width = h.shape
                h_flat = h.view(batch_size, channels, -1).transpose(1, 2)
                h_flat = layer(h_flat, context)
                h = h_flat.transpose(1, 2).view(batch_size, channels, height, width)
        
        for block in self.up_blocks:
            for layer in block:
                if isinstance(layer, ResNetBlock):
                    skip = skip_connections.pop()
                    h = torch.cat([h, skip], dim=1)
                    h = layer(h, time_emb)
                elif isinstance(layer, CrossAttentionBlock) and context is not None:
                    batch_size, channels, height, width = h.shape
                    h_flat = h.view(batch_size, channels, -1).transpose(1, 2)
                    h_flat = layer(h_flat, context)
                    h = h_flat.transpose(1, 2).view(batch_size, channels, height, width)
                else:
                    h = layer(h)
        
        return self.output_conv(h)

class DiffusionImageGenerator(nn.Module):
    """Diffusion-based image generator with advanced conditioning."""
    
    def __init__(self, args: DiffusionImageGeneratorArgs):
        super().__init__()
        self.args = args
        
        self.unet = UNetModel(args)
        
        if args.enable_brand_conditioning:
            self.brand_encoder = nn.Sequential(
                nn.Linear(768, 768),
                nn.LayerNorm(768),
                nn.ReLU(),
                nn.Linear(768, 768)
            )
        
        self.noise_scheduler = DDPMScheduler(args.num_inference_steps)
        
    def forward(self, noisy_images: torch.Tensor, timesteps: torch.Tensor, 
                brand_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        context = None
        if self.args.enable_brand_conditioning and brand_context is not None:
            context = self.brand_encoder(brand_context).unsqueeze(1)
        
        return self.unet(noisy_images, timesteps, context)
    
    def generate(self, batch_size: int = 1, brand_context: Optional[torch.Tensor] = None,
                guidance_scale: Optional[float] = None) -> torch.Tensor:
        """Generate images using diffusion process."""
        if guidance_scale is None:
            guidance_scale = self.args.guidance_scale
        
        device = next(self.parameters()).device
        
        images = torch.randn(
            batch_size, self.args.num_channels, self.args.image_size, self.args.image_size,
            device=device
        )
        
        for t in reversed(range(self.args.num_inference_steps)):
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            with torch.no_grad():
                noise_pred = self.forward(images, timesteps, brand_context)
                
                if guidance_scale > 1.0 and brand_context is not None:
                    noise_pred_uncond = self.forward(images, timesteps, None)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
                
                images = self.noise_scheduler.step(noise_pred, t, images)
        
        return images

class DDPMScheduler:
    """DDPM noise scheduler for diffusion process."""
    
    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_train_timesteps = num_train_timesteps
        
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor) -> torch.Tensor:
        """Perform one denoising step."""
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[timestep - 1] if timestep > 0 else torch.tensor(1.0)
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        return prev_sample

class BrandConsistentImageGenerator:
    """Wrapper for brand-consistent image generation."""
    
    def __init__(self, model: DiffusionImageGenerator):
        self.model = model
        
    def generate_brand_images(self, brand_profile: torch.Tensor, content_type: str = 'social_post',
                            num_images: int = 1) -> torch.Tensor:
        """Generate images consistent with brand profile."""
        return self.model.generate(
            batch_size=num_images,
            brand_context=brand_profile
        )
    
    def generate_layout_variations(self, base_layout: torch.Tensor, brand_profile: torch.Tensor,
                                 num_variations: int = 4) -> torch.Tensor:
        """Generate layout variations for given brand."""
        variations = []
        for _ in range(num_variations):
            variation = self.model.generate(
                batch_size=1,
                brand_context=brand_profile
            )
            variations.append(variation)
        
        return torch.cat(variations, dim=0)

class LayoutGenerator(nn.Module):
    """Generator for layout and composition elements."""
    
    def __init__(self, hidden_size: int = 512):
        super().__init__()
        
        self.layout_encoder = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        self.composition_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # x, y, width, height
        )
        
        self.element_type_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # Different element types
        )
        
    def forward(self, brand_context: torch.Tensor) -> Dict[str, torch.Tensor]:
        layout_features = self.layout_encoder(brand_context)
        
        composition = self.composition_head(layout_features)
        element_types = self.element_type_head(layout_features)
        
        return {
            'composition': composition,
            'element_types': element_types,
            'layout_features': layout_features
        }

def create_enhanced_image_generator(config: Dict[str, Any]) -> DiffusionImageGenerator:
    """Create enhanced image generator from configuration."""
    args = DiffusionImageGeneratorArgs(
        image_size=config.get('image_size', 512),
        num_inference_steps=config.get('num_inference_steps', 50),
        guidance_scale=config.get('guidance_scale', 7.5),
        num_channels=config.get('num_channels', 3),
        unet_channels=config.get('unet_channels', [320, 640, 1280, 1280]),
        unet_layers=config.get('unet_layers', [2, 2, 2, 2]),
        attention_resolutions=config.get('attention_resolutions', [4, 2, 1]),
        enable_brand_conditioning=config.get('enable_brand_conditioning', True),
        enable_layout_generation=config.get('enable_layout_generation', True),
        enable_style_transfer=config.get('enable_style_transfer', True)
    )
    
    return DiffusionImageGenerator(args)
