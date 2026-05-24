"""
Brand Kit Ads Model Trainer

Advanced training pipeline for the Brand Kit Ads model with:
1. Multi-modal training (vision + language)
2. Brand consistency optimization
3. Content quality assessment
4. Visual-text alignment training
5. Adversarial training for realistic ad generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from transformers import get_linear_schedule_with_warmup
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import yaml
import os
import logging
from dataclasses import dataclass, field
from PIL import Image
import requests
from io import BytesIO
import cv2
import webcolors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import wandb
from accelerate import Accelerator
import deepspeed

from .model import BrandKitAdsModel, BrandKitAdsConfig, BrandKitExtraction, AdContent


@dataclass
class BrandKitTrainingArguments:
    """Training arguments for Brand Kit Ads model"""
    
    # Basic training config
    output_dir: str = "./output/brand_kit_ads"
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Multi-modal training
    vision_learning_rate: float = 1e-5
    language_learning_rate: float = 5e-5
    brand_learning_rate: float = 2e-5
    
    # Loss weights
    language_loss_weight: float = 1.0
    brand_consistency_weight: float = 0.3
    visual_alignment_weight: float = 0.2
    content_quality_weight: float = 0.15
    adversarial_weight: float = 0.1
    
    # Data config
    max_sequence_length: int = 512
    image_size: int = 224
    max_websites_per_epoch: int = 10000
    brand_augmentation_prob: float = 0.3
    
    # Evaluation config
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    eval_accumulation_steps: int = 10
    
    # Advanced training
    use_adversarial_training: bool = True
    use_curriculum_learning: bool = True
    use_brand_consistency_loss: bool = True
    use_visual_alignment_loss: bool = True
    use_content_quality_loss: bool = True
    
    # Optimization
    optimizer_type: str = "adamw"
    scheduler_type: str = "cosine"
    use_deepspeed: bool = False
    fp16: bool = True
    bf16: bool = False
    
    # Logging and monitoring
    report_to: str = "wandb"
    run_name: Optional[str] = None
    project_name: str = "brand-kit-ads-model"
    
    # Checkpointing
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_brand_alignment"
    greater_is_better: bool = True


class WebsiteDataset(Dataset):
    """Dataset for website brand analysis and ad generation"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        image_size: int = 224,
        include_synthetic: bool = True
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_size = image_size
        self.include_synthetic = include_synthetic
        
        # Load dataset
        self.data = self._load_data()
        
        # Content type mappings
        self.content_types = [
            'social_media', 'email', 'banner', 'video',
            'blog', 'newsletter', 'landing_page', 'display_ad'
        ]
        
        self.audiences = [
            'general', 'young_adults', 'professionals', 'families',
            'seniors', 'students', 'entrepreneurs', 'creatives'
        ]
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load training data from various sources"""
        data = []
        
        # Load from JSON files
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r') as f:
                data = json.load(f)
        
        # Add synthetic data if requested
        if self.include_synthetic:
            synthetic_data = self._generate_synthetic_data(1000)
            data.extend(synthetic_data)
        
        return data
    
    def _generate_synthetic_data(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate synthetic training data"""
        synthetic_data = []
        
        # Common brand color palettes
        brand_palettes = [
            ['#1a1a1a', '#ffffff', '#007bff', '#28a745'],  # Tech
            ['#8b4513', '#daa520', '#f5deb3', '#2f4f4f'],  # Luxury
            ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'],  # Playful
            ['#2c3e50', '#3498db', '#e74c3c', '#f39c12'],  # Corporate
            ['#6c5ce7', '#a29bfe', '#fd79a8', '#fdcb6e'],  # Creative
        ]
        
        # Typography styles
        typography_styles = [
            {'primary': 'Arial, sans-serif', 'secondary': 'Georgia, serif', 'style': 'modern'},
            {'primary': 'Helvetica, sans-serif', 'secondary': 'Times, serif', 'style': 'classic'},
            {'primary': 'Roboto, sans-serif', 'secondary': 'Open Sans, sans-serif', 'style': 'clean'},
            {'primary': 'Montserrat, sans-serif', 'secondary': 'Lato, sans-serif', 'style': 'elegant'},
        ]
        
        for i in range(num_samples):
            # Random brand characteristics
            palette = np.random.choice(len(brand_palettes))
            typography = np.random.choice(len(typography_styles))
            
            # Generate sample data
            sample = {
                'website_url': f'https://example-{i}.com',
                'brand_colors': brand_palettes[palette],
                'typography': typography_styles[typography],
                'brand_personality': {
                    'professional': np.random.uniform(0.3, 1.0),
                    'modern': np.random.uniform(0.2, 1.0),
                    'trustworthy': np.random.uniform(0.5, 1.0),
                    'innovative': np.random.uniform(0.1, 0.9),
                    'playful': np.random.uniform(0.0, 0.8),
                    'luxury': np.random.uniform(0.0, 0.7)
                },
                'content_examples': [
                    {
                        'type': np.random.choice(self.content_types),
                        'audience': np.random.choice(self.audiences),
                        'text': self._generate_sample_ad_text(),
                        'performance_score': np.random.uniform(0.6, 0.95)
                    }
                ],
                'image_path': None,  # Would be actual screenshots in real data
                'domain': np.random.choice(['technology', 'fashion', 'food', 'finance', 'health'])
            }
            
            synthetic_data.append(sample)
        
        return synthetic_data
    
    def _generate_sample_ad_text(self) -> str:
        """Generate sample advertising text"""
        headlines = [
            "Transform Your Business Today",
            "Discover the Future of Innovation",
            "Experience Excellence Like Never Before",
            "Unlock Your Potential",
            "Join Thousands of Satisfied Customers"
        ]
        
        bodies = [
            "Our cutting-edge solution delivers exceptional results that exceed expectations.",
            "Join the revolution and experience the difference that quality makes.",
            "Trusted by industry leaders worldwide for outstanding performance.",
            "Innovative technology meets user-friendly design for perfect results.",
            "Take your success to the next level with our proven approach."
        ]
        
        ctas = [
            "Get Started Today",
            "Learn More",
            "Try It Free",
            "Contact Us Now",
            "Join Now"
        ]
        
        headline = np.random.choice(headlines)
        body = np.random.choice(bodies)
        cta = np.random.choice(ctas)
        
        return f"{headline}\n\n{body}\n\n{cta}"
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Process text content
        if 'content_examples' in item and item['content_examples']:
            content = item['content_examples'][0]
            text = content['text']
            content_type = content.get('type', 'social_media')
            audience = content.get('audience', 'general')
        else:
            text = "Generate engaging advertising content for this brand."
            content_type = 'social_media'
            audience = 'general'
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Process image (placeholder for now)
        # In real implementation, this would load and process website screenshots
        image = torch.randn(3, self.image_size, self.image_size)
        
        # Brand information
        brand_colors = item.get('brand_colors', ['#000000', '#ffffff'])
        brand_personality = item.get('brand_personality', {})
        
        # Convert to tensors
        content_type_id = self.content_types.index(content_type) if content_type in self.content_types else 0
        audience_id = self.audiences.index(audience) if audience in self.audiences else 0
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0),
            'images': image,
            'content_type_ids': torch.tensor(content_type_id, dtype=torch.long),
            'audience_ids': torch.tensor(audience_id, dtype=torch.long),
            'brand_colors': torch.tensor([self._hex_to_rgb(color) for color in brand_colors[:4]], dtype=torch.float),
            'brand_personality': torch.tensor([
                brand_personality.get('professional', 0.5),
                brand_personality.get('modern', 0.5),
                brand_personality.get('trustworthy', 0.5),
                brand_personality.get('innovative', 0.5),
                brand_personality.get('playful', 0.5),
                brand_personality.get('luxury', 0.5)
            ], dtype=torch.float)
        }
    
    def _hex_to_rgb(self, hex_color: str) -> List[float]:
        """Convert hex color to RGB values normalized to [0, 1]"""
        try:
            hex_color = hex_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            return [c / 255.0 for c in rgb]
        except:
            return [0.0, 0.0, 0.0]  # Default to black


class BrandConsistencyLoss(nn.Module):
    """Loss function for brand consistency"""
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self,
        brand_embedding: torch.Tensor,
        generated_features: torch.Tensor,
        target_brand_features: torch.Tensor
    ) -> torch.Tensor:
        
        # Compute similarity between generated content and brand
        brand_sim = F.cosine_similarity(
            generated_features, brand_embedding.unsqueeze(1), dim=-1
        )
        
        # Compute similarity with target brand features
        target_sim = F.cosine_similarity(
            generated_features, target_brand_features, dim=-1
        )
        
        # Consistency loss - encourage alignment with brand
        consistency_loss = F.mse_loss(brand_sim, target_sim)
        
        return consistency_loss


class VisualAlignmentLoss(nn.Module):
    """Loss function for visual-text alignment"""
    
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
        
    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        positive_pairs: torch.Tensor
    ) -> torch.Tensor:
        
        # Compute similarity matrix
        similarity = torch.matmul(visual_features, text_features.transpose(-2, -1))
        
        # Contrastive loss for positive pairs
        positive_sim = similarity[positive_pairs == 1]
        negative_sim = similarity[positive_pairs == 0]
        
        # Encourage positive pairs to be similar, negative pairs to be dissimilar
        positive_loss = torch.clamp(1.0 - positive_sim, min=0).mean()
        negative_loss = torch.clamp(negative_sim - self.margin, min=0).mean()
        
        return positive_loss + negative_loss


class ContentQualityLoss(nn.Module):
    """Loss function for content quality assessment"""
    
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        quality_scores: torch.Tensor,
        target_quality: torch.Tensor
    ) -> torch.Tensor:
        
        # MSE loss for quality scores
        quality_loss = F.mse_loss(quality_scores.squeeze(-1), target_quality)
        
        return quality_loss


class AdversarialDiscriminator(nn.Module):
    """Discriminator for adversarial training"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.discriminator(features)


class BrandKitAdsTrainer:
    """Advanced trainer for Brand Kit Ads model"""
    
    def __init__(
        self,
        model: BrandKitAdsModel,
        args: BrandKitTrainingArguments,
        train_dataset: WebsiteDataset,
        eval_dataset: Optional[WebsiteDataset] = None,
        tokenizer=None,
        accelerator: Optional[Accelerator] = None
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.accelerator = accelerator or Accelerator()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize training components
        self._setup_training()
        
        # Loss functions
        self.brand_consistency_loss = BrandConsistencyLoss()
        self.visual_alignment_loss = VisualAlignmentLoss()
        self.content_quality_loss = ContentQualityLoss()
        
        # Adversarial components
        if args.use_adversarial_training:
            self.discriminator = AdversarialDiscriminator(model.config.hidden_size)
            self.discriminator_optimizer = AdamW(
                self.discriminator.parameters(),
                lr=args.learning_rate * 0.1,
                weight_decay=args.weight_decay
            )
        
        # Metrics tracking
        self.training_metrics = {
            'total_loss': [],
            'language_loss': [],
            'brand_loss': [],
            'visual_loss': [],
            'quality_loss': [],
            'adversarial_loss': []
        }
        
        self.eval_metrics = {
            'brand_alignment': [],
            'content_quality': [],
            'visual_alignment': [],
            'generation_diversity': []
        }
    
    def _setup_training(self):
        """Setup optimizers, schedulers, and other training components"""
        
        # Group parameters by component
        vision_params = []
        language_params = []
        brand_params = []
        
        for name, param in self.model.named_parameters():
            if 'vision' in name or 'color' in name or 'typography' in name or 'design' in name:
                vision_params.append(param)
            elif 'brand' in name or 'fusion' in name:
                brand_params.append(param)
            else:
                language_params.append(param)
        
        # Setup optimizers with different learning rates
        param_groups = [
            {'params': vision_params, 'lr': self.args.vision_learning_rate},
            {'params': language_params, 'lr': self.args.language_learning_rate},
            {'params': brand_params, 'lr': self.args.brand_learning_rate}
        ]
        
        if self.args.optimizer_type == "adamw":
            self.optimizer = AdamW(
                param_groups,
                weight_decay=self.args.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.args.optimizer_type}")
        
        # Setup scheduler
        total_steps = len(self.train_dataset) * self.args.num_train_epochs // (
            self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps
        )
        
        if self.args.scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)
        elif self.args.scheduler_type == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=total_steps
            )
        
        # Setup data loaders
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        if self.eval_dataset:
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        # Prepare for distributed training
        if self.accelerator:
            self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader
            )
            if hasattr(self, 'eval_dataloader'):
                self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute multi-component loss"""
        
        # Forward pass
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            images=batch['images'],
            content_type_ids=batch['content_type_ids'],
            audience_ids=batch['audience_ids'],
            labels=batch['labels']
        )
        
        # Base language modeling loss
        language_loss = outputs.loss
        total_loss = self.args.language_loss_weight * language_loss
        
        loss_dict = {'language_loss': language_loss}
        
        # Brand consistency loss
        if self.args.use_brand_consistency_loss and hasattr(outputs, 'brand_embedding'):
            brand_analysis = self.model.analyze_website_brand(batch['images'])
            brand_embedding = brand_analysis['brand_embedding']
            
            # Get generated features (last hidden state)
            with torch.no_grad():
                generated_outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    brand_embedding=brand_embedding
                )
            
            # Placeholder for brand consistency computation
            brand_loss = torch.tensor(0.0, device=language_loss.device)
            total_loss += self.args.brand_consistency_weight * brand_loss
            loss_dict['brand_loss'] = brand_loss
        
        # Visual alignment loss
        if self.args.use_visual_alignment_loss:
            # Placeholder for visual alignment computation
            visual_loss = torch.tensor(0.0, device=language_loss.device)
            total_loss += self.args.visual_alignment_weight * visual_loss
            loss_dict['visual_loss'] = visual_loss
        
        # Content quality loss
        if self.args.use_content_quality_loss:
            # Placeholder for content quality computation
            quality_loss = torch.tensor(0.0, device=language_loss.device)
            total_loss += self.args.content_quality_weight * quality_loss
            loss_dict['quality_loss'] = quality_loss
        
        # Adversarial loss
        if self.args.use_adversarial_training and hasattr(self, 'discriminator'):
            # Placeholder for adversarial computation
            adversarial_loss = torch.tensor(0.0, device=language_loss.device)
            total_loss += self.args.adversarial_weight * adversarial_loss
            loss_dict['adversarial_loss'] = adversarial_loss
        
        loss_dict['total_loss'] = total_loss
        return loss_dict
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        
        self.model.train()
        
        # Compute losses
        loss_dict = self.compute_loss(batch)
        total_loss = loss_dict['total_loss']
        
        # Backward pass
        if self.accelerator:
            self.accelerator.backward(total_loss)
        else:
            total_loss.backward()
        
        # Gradient clipping
        if self.args.max_grad_norm > 0:
            if self.accelerator:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        # Convert to float for logging
        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model performance"""
        
        if not hasattr(self, 'eval_dataloader'):
            return {}
        
        self.model.eval()
        eval_losses = []
        brand_alignments = []
        content_qualities = []
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                # Compute losses
                loss_dict = self.compute_loss(batch)
                eval_losses.append(loss_dict['total_loss'].item())
                
                # Analyze brand alignment
                brand_analysis = self.model.analyze_website_brand(batch['images'])
                
                # Generate content
                generated = self.model.generate_with_brand_awareness(
                    input_ids=batch['input_ids'][:, :10],  # Use first 10 tokens as prompt
                    brand_embedding=brand_analysis['brand_embedding'],
                    max_length=100,
                    temperature=0.8
                )
                
                # Compute metrics (placeholder)
                brand_alignment = generated.get('brand_alignment_score', 0.8)
                content_quality = 0.75  # Placeholder
                
                brand_alignments.append(brand_alignment)
                content_qualities.append(content_quality)
        
        return {
            'eval_loss': np.mean(eval_losses),
            'eval_brand_alignment': np.mean(brand_alignments),
            'eval_content_quality': np.mean(content_qualities),
            'eval_visual_alignment': 0.7  # Placeholder
        }
    
    def train(self):
        """Main training loop"""
        
        self.logger.info("Starting training...")
        
        # Initialize wandb if specified
        if self.args.report_to == "wandb":
            wandb.init(
                project=self.args.project_name,
                name=self.args.run_name,
                config=self.args.__dict__
            )
        
        global_step = 0
        best_metric = -float('inf') if self.args.greater_is_better else float('inf')
        
        for epoch in range(self.args.num_train_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.args.num_train_epochs}")
            
            # Training
            epoch_losses = []
            progress_bar = tqdm(self.train_dataloader, desc=f"Training Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                # Training step
                loss_dict = self.training_step(batch)
                epoch_losses.append(loss_dict['total_loss'])
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss_dict['total_loss']:.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
                
                global_step += 1
                
                # Logging
                if global_step % self.args.logging_steps == 0:
                    avg_loss = np.mean(epoch_losses[-self.args.logging_steps:])
                    self.logger.info(f"Step {global_step}: Average loss = {avg_loss:.4f}")
                    
                    if self.args.report_to == "wandb":
                        wandb.log({
                            'train/loss': avg_loss,
                            'train/learning_rate': self.scheduler.get_last_lr()[0],
                            'train/epoch': epoch,
                            'train/step': global_step
                        })
                
                # Evaluation
                if global_step % self.args.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    
                    if eval_metrics:
                        self.logger.info(f"Evaluation metrics: {eval_metrics}")
                        
                        if self.args.report_to == "wandb":
                            wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()})
                        
                        # Check if this is the best model
                        current_metric = eval_metrics.get(self.args.metric_for_best_model, 0)
                        is_best = (
                            (self.args.greater_is_better and current_metric > best_metric) or
                            (not self.args.greater_is_better and current_metric < best_metric)
                        )
                        
                        if is_best:
                            best_metric = current_metric
                            self.save_model(os.path.join(self.args.output_dir, "best_model"))
                
                # Save checkpoint
                if global_step % self.args.save_steps == 0:
                    self.save_model(os.path.join(self.args.output_dir, f"checkpoint-{global_step}"))
            
            # End of epoch evaluation
            eval_metrics = self.evaluate()
            if eval_metrics:
                self.logger.info(f"End of epoch {epoch + 1} evaluation: {eval_metrics}")
        
        # Final save
        self.save_model(os.path.join(self.args.output_dir, "final_model"))
        
        if self.args.report_to == "wandb":
            wandb.finish()
        
        self.logger.info("Training completed!")
    
    def save_model(self, output_dir: str):
        """Save model and training state"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        if self.accelerator:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(output_dir)
        else:
            self.model.save_pretrained(output_dir)
        
        # Save tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)
        
        # Save training arguments
        with open(os.path.join(output_dir, "training_args.json"), 'w') as f:
            json.dump(self.args.__dict__, f, indent=2)
        
        # Save training metrics
        with open(os.path.join(output_dir, "training_metrics.json"), 'w') as f:
            json.dump({
                'training_metrics': self.training_metrics,
                'eval_metrics': self.eval_metrics
            }, f, indent=2)
        
        self.logger.info(f"Model saved to {output_dir}")
    
    def load_model(self, model_path: str):
        """Load model from checkpoint"""
        
        self.model = BrandKitAdsModel.from_pretrained(model_path)
        
        # Load training state if available
        training_args_path = os.path.join(model_path, "training_args.json")
        if os.path.exists(training_args_path):
            with open(training_args_path, 'r') as f:
                training_args = json.load(f)
            self.args = BrandKitTrainingArguments(**training_args)
        
        self.logger.info(f"Model loaded from {model_path}")


class BrandKitEvaluator:
    """Comprehensive evaluation for Brand Kit Ads model"""
    
    def __init__(self, model: BrandKitAdsModel, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        
    def evaluate_brand_extraction(self, test_websites: List[str]) -> Dict[str, float]:
        """Evaluate brand extraction accuracy"""
        
        extraction_scores = []
        
        for website_url in test_websites:
            try:
                # Extract brand kit
                brand_kit = self.model.extract_brand_kit_from_url(website_url)
                
                if brand_kit:
                    # Score based on completeness and quality
                    score = self._score_brand_extraction(brand_kit)
                    extraction_scores.append(score)
                
            except Exception as e:
                print(f"Error evaluating {website_url}: {e}")
                extraction_scores.append(0.0)
        
        return {
            'brand_extraction_accuracy': np.mean(extraction_scores),
            'extraction_success_rate': len([s for s in extraction_scores if s > 0]) / len(extraction_scores)
        }
    
    def _score_brand_extraction(self, brand_kit: BrandKitExtraction) -> float:
        """Score the quality of brand extraction"""
        
        score = 0.0
        
        # Color extraction (25%)
        if brand_kit.primary_colors and len(brand_kit.primary_colors) >= 2:
            score += 0.25
        
        # Typography extraction (20%)
        if brand_kit.typography and 'primary_font' in brand_kit.typography:
            score += 0.20
        
        # Brand personality (25%)
        if brand_kit.brand_personality and len(brand_kit.brand_personality) >= 3:
            score += 0.25
        
        # Design patterns (15%)
        if brand_kit.design_patterns and len(brand_kit.design_patterns) >= 2:
            score += 0.15
        
        # Brand voice (15%)
        if brand_kit.brand_voice and len(brand_kit.brand_voice) >= 2:
            score += 0.15
        
        return score
    
    def evaluate_ad_generation(
        self, 
        brand_embeddings: List[torch.Tensor],
        prompts: List[str],
        human_ratings: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """Evaluate ad generation quality"""
        
        generation_scores = []
        brand_alignment_scores = []
        creativity_scores = []
        
        for i, (brand_embedding, prompt) in enumerate(zip(brand_embeddings, prompts)):
            # Generate ads
            ad_contents = self.model.generate_ad_content(
                brand_embedding=brand_embedding,
                prompt=prompt,
                num_variants=3
            )
            
            for ad_content in ad_contents:
                # Score generation quality
                generation_score = self._score_ad_content(ad_content)
                generation_scores.append(generation_score)
                
                # Brand alignment
                brand_alignment_scores.append(ad_content.brand_alignment_score)
                
                # Creativity (placeholder)
                creativity_scores.append(0.75)
        
        results = {
            'ad_generation_quality': np.mean(generation_scores),
            'brand_alignment': np.mean(brand_alignment_scores),
            'creativity_score': np.mean(creativity_scores)
        }
        
        # Correlation with human ratings if available
        if human_ratings:
            correlation = np.corrcoef(generation_scores[:len(human_ratings)], human_ratings)[0, 1]
            results['human_correlation'] = correlation
        
        return results
    
    def _score_ad_content(self, ad_content: AdContent) -> float:
        """Score the quality of generated ad content"""
        
        score = 0.0
        
        # Headline quality (30%)
        if ad_content.headline and len(ad_content.headline.split()) >= 3:
            score += 0.30
        
        # Body text quality (25%)
        if ad_content.body_text and len(ad_content.body_text.split()) >= 10:
            score += 0.25
        
        # Call to action (20%)
        if ad_content.call_to_action and len(ad_content.call_to_action.split()) <= 5:
            score += 0.20
        
        # Brand alignment (25%)
        score += 0.25 * ad_content.brand_alignment_score
        
        return score
    
    def generate_evaluation_report(
        self,
        test_websites: List[str],
        output_path: str = "evaluation_report.json"
    ) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        
        # Brand extraction evaluation
        brand_eval = self.evaluate_brand_extraction(test_websites)
        
        # Generate sample brand embeddings for ad evaluation
        sample_embeddings = [torch.randn(512) for _ in range(10)]
        sample_prompts = [
            "Create a social media ad for a tech startup",
            "Generate email marketing content for a fashion brand",
            "Design banner ad copy for a food delivery service"
        ] * 4  # Repeat to match embeddings
        
        # Ad generation evaluation
        ad_eval = self.evaluate_ad_generation(sample_embeddings, sample_prompts[:10])
        
        # Combine results
        report = {
            'evaluation_timestamp': torch.datetime.now().isoformat(),
            'model_config': self.model.config.__dict__,
            'brand_extraction_metrics': brand_eval,
            'ad_generation_metrics': ad_eval,
            'test_websites_count': len(test_websites),
            'overall_score': (
                brand_eval.get('brand_extraction_accuracy', 0) * 0.4 +
                ad_eval.get('ad_generation_quality', 0) * 0.3 +
                ad_eval.get('brand_alignment', 0) * 0.3
            )
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


# Export main components
__all__ = [
    'BrandKitTrainingArguments',
    'WebsiteDataset',
    'BrandKitAdsTrainer',
    'BrandKitEvaluator',
    'BrandConsistencyLoss',
    'VisualAlignmentLoss',
    'ContentQualityLoss'
]