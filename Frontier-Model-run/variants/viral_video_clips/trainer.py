"""
Viral Video Clips Model - Advanced Training Pipeline

This module provides a comprehensive training pipeline for the Viral Video Clips model,
including multi-task learning, curriculum learning, adversarial training, and 
specialized loss functions for viral content optimization.

Key Features:
- Multi-task learning (video understanding, highlight detection, caption generation)
- Curriculum learning with progressive difficulty
- Adversarial training for realistic content generation
- Contrastive learning for better feature representations
- Viral pattern recognition and engagement prediction
- Platform-specific optimization
- Real-time performance monitoring and evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR
import torch.distributed as dist
from transformers import get_scheduler
from accelerate import Accelerator
import deepspeed

import numpy as np
import pandas as pd
import json
import yaml
import os
import time
import logging
import wandb
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import cv2
import librosa
import moviepy.editor as mp
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
import aiohttp
from collections import defaultdict
import pickle
import hashlib

from model import ViralVideoClipsModel, ViralVideoClipsConfig, VideoClip, VideoAnalysis


@dataclass
class ViralVideoTrainingArguments:
    """Training arguments for Viral Video Clips model"""
    
    # Basic training settings
    output_dir: str = "./output/viral_video_clips"
    num_train_epochs: int = 15
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Multi-modal learning rates
    video_learning_rate: float = 1e-5
    audio_learning_rate: float = 2e-5
    caption_learning_rate: float = 5e-5
    effects_learning_rate: float = 3e-5
    
    # Loss weights
    video_understanding_weight: float = 1.0
    highlight_detection_weight: float = 0.8
    caption_generation_weight: float = 0.6
    viral_prediction_weight: float = 0.4
    engagement_prediction_weight: float = 0.3
    contrastive_loss_weight: float = 0.2
    adversarial_loss_weight: float = 0.1
    
    # Advanced training features
    use_curriculum_learning: bool = True
    use_adversarial_training: bool = True
    use_contrastive_learning: bool = True
    use_multi_task_learning: bool = True
    
    # Optimization settings
    optimizer_type: str = "adamw"
    scheduler_type: str = "cosine_with_restarts"
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_deepspeed: bool = False
    
    # Evaluation and logging
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "viral_prediction_accuracy"
    greater_is_better: bool = True
    
    # Monitoring
    report_to: str = "wandb"
    project_name: str = "viral-video-clips-model"
    run_name: Optional[str] = None
    
    # Data augmentation
    use_video_augmentation: bool = True
    use_audio_augmentation: bool = True
    augmentation_probability: float = 0.3
    
    # Hardware optimization
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    max_memory_usage: float = 0.9


class ViralVideoDataset(Dataset):
    """Dataset for viral video clips training"""
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        config: ViralVideoClipsConfig = None,
        split: str = "train",
        max_samples: Optional[int] = None,
        include_synthetic: bool = True
    ):
        self.config = config or ViralVideoClipsConfig()
        self.split = split
        self.max_samples = max_samples
        self.include_synthetic = include_synthetic
        
        # Load data
        self.data = self._load_data(data_path)
        
        # Apply sample limit
        if max_samples and len(self.data) > max_samples:
            self.data = self.data[:max_samples]
        
        # Create synthetic data if needed
        if include_synthetic and len(self.data) < 100:
            synthetic_data = self._create_synthetic_data(max(100, len(self.data)))
            self.data.extend(synthetic_data)
        
        logging.info(f"Loaded {len(self.data)} samples for {split} split")
    
    def _load_data(self, data_path: Optional[str]) -> List[Dict[str, Any]]:
        """Load training data from file or create synthetic data"""
        
        if data_path and os.path.exists(data_path):
            with open(data_path, 'r') as f:
                data = json.load(f)
            return data
        else:
            # Create synthetic data for demonstration
            return self._create_synthetic_data(500)
    
    def _create_synthetic_data(self, num_samples: int) -> List[Dict[str, Any]]:
        """Create synthetic training data"""
        
        synthetic_data = []
        
        # Viral content categories
        categories = [
            "comedy", "dance", "tutorial", "reaction", "challenge",
            "lifestyle", "food", "travel", "tech", "sports"
        ]
        
        # Viral patterns
        viral_patterns = [
            "hook_in_first_3_seconds",
            "emotional_peak_mid_video",
            "surprise_ending",
            "trending_audio",
            "face_close_up",
            "quick_cuts",
            "text_overlay",
            "before_after",
            "transformation",
            "call_to_action"
        ]
        
        for i in range(num_samples):
            # Generate synthetic video metadata
            duration = np.random.uniform(15, 300)  # 15 seconds to 5 minutes
            category = np.random.choice(categories)
            
            # Generate viral metrics (simulated)
            views = np.random.lognormal(10, 2)  # Log-normal distribution for views
            likes = views * np.random.uniform(0.01, 0.1)
            shares = likes * np.random.uniform(0.1, 0.3)
            comments = likes * np.random.uniform(0.05, 0.2)
            
            # Calculate viral score
            engagement_rate = (likes + shares + comments) / max(views, 1)
            viral_score = min(1.0, engagement_rate * 10)
            
            # Generate highlight moments
            num_highlights = np.random.randint(1, 6)
            highlights = []
            for j in range(num_highlights):
                start = np.random.uniform(0, duration - 10)
                end = min(duration, start + np.random.uniform(5, 30))
                highlights.append({
                    'start': start,
                    'end': end,
                    'score': np.random.uniform(0.3, 1.0),
                    'type': np.random.choice(['peak_action', 'emotional_moment', 'surprise', 'climax'])
                })
            
            # Generate captions
            num_captions = np.random.randint(0, 8)
            captions = []
            for j in range(num_captions):
                start = np.random.uniform(0, duration - 5)
                end = min(duration, start + np.random.uniform(2, 8))
                captions.append({
                    'start': start,
                    'end': end,
                    'text': f"Caption {j+1} for {category} content",
                    'confidence': np.random.uniform(0.7, 1.0)
                })
            
            # Generate platform performance
            platform_performance = {
                'tiktok': {
                    'views': views * np.random.uniform(0.3, 0.5),
                    'engagement_rate': np.random.uniform(0.02, 0.15),
                    'completion_rate': np.random.uniform(0.4, 0.9)
                },
                'instagram': {
                    'views': views * np.random.uniform(0.2, 0.4),
                    'engagement_rate': np.random.uniform(0.01, 0.08),
                    'completion_rate': np.random.uniform(0.3, 0.8)
                },
                'youtube_shorts': {
                    'views': views * np.random.uniform(0.1, 0.3),
                    'engagement_rate': np.random.uniform(0.005, 0.05),
                    'completion_rate': np.random.uniform(0.2, 0.7)
                }
            }
            
            sample = {
                'video_id': f"synthetic_{i:06d}",
                'title': f"Viral {category} content #{i+1}",
                'duration': duration,
                'category': category,
                'viral_score': viral_score,
                'engagement_metrics': {
                    'views': int(views),
                    'likes': int(likes),
                    'shares': int(shares),
                    'comments': int(comments),
                    'engagement_rate': engagement_rate
                },
                'highlights': highlights,
                'captions': captions,
                'viral_patterns': np.random.choice(viral_patterns, size=np.random.randint(1, 4), replace=False).tolist(),
                'platform_performance': platform_performance,
                'audio_features': {
                    'has_speech': np.random.choice([True, False], p=[0.7, 0.3]),
                    'has_music': np.random.choice([True, False], p=[0.8, 0.2]),
                    'audio_quality': np.random.uniform(0.5, 1.0),
                    'volume_consistency': np.random.uniform(0.6, 1.0)
                },
                'visual_features': {
                    'resolution': np.random.choice([(720, 1280), (1080, 1920)]),
                    'fps': np.random.choice([24, 30, 60]),
                    'brightness': np.random.uniform(0.3, 0.9),
                    'contrast': np.random.uniform(0.4, 1.0),
                    'saturation': np.random.uniform(0.5, 1.2),
                    'motion_intensity': np.random.uniform(0.1, 1.0)
                },
                'trending_topics': np.random.choice([
                    'ai', 'technology', 'lifestyle', 'fitness', 'food',
                    'travel', 'education', 'entertainment', 'news', 'sports'
                ], size=np.random.randint(1, 4), replace=False).tolist(),
                'upload_timestamp': (datetime.now() - timedelta(days=np.random.randint(1, 365))).isoformat(),
                'creator_metrics': {
                    'followers': np.random.lognormal(8, 2),
                    'avg_views': np.random.lognormal(9, 1.5),
                    'engagement_rate': np.random.uniform(0.01, 0.1)
                }
            }
            
            synthetic_data.append(sample)
        
        return synthetic_data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a training sample"""
        
        sample = self.data[idx]
        
        # Create synthetic video and audio tensors
        # In practice, these would be loaded from actual video files
        video_frames = torch.randn(3, 16, 224, 224)  # (C, T, H, W)
        audio_waveform = torch.randn(16000 * 10)  # 10 seconds of audio
        
        # Create target tensors
        viral_score = torch.tensor(sample['viral_score'], dtype=torch.float32)
        engagement_rate = torch.tensor(sample['engagement_metrics']['engagement_rate'], dtype=torch.float32)
        
        # Scene change labels (synthetic)
        scene_changes = torch.zeros(16)  # 16 time steps
        if sample['highlights']:
            for highlight in sample['highlights']:
                start_frame = int((highlight['start'] / sample['duration']) * 16)
                end_frame = int((highlight['end'] / sample['duration']) * 16)
                scene_changes[start_frame:end_frame] = highlight['score']
        
        # Highlight labels
        highlight_labels = scene_changes.clone()
        
        # Caption tokens (simplified)
        caption_text = " ".join([cap['text'] for cap in sample['captions']])
        caption_tokens = torch.randint(0, 1000, (50,))  # Simplified tokenization
        
        # Platform optimization targets
        platform_targets = torch.tensor([
            sample['platform_performance']['tiktok']['engagement_rate'],
            sample['platform_performance']['instagram']['engagement_rate'],
            sample['platform_performance']['youtube_shorts']['engagement_rate']
        ], dtype=torch.float32)
        
        return {
            'video_frames': video_frames,
            'audio_waveform': audio_waveform,
            'viral_score': viral_score,
            'engagement_rate': engagement_rate,
            'scene_changes': scene_changes,
            'highlight_labels': highlight_labels,
            'caption_tokens': caption_tokens,
            'platform_targets': platform_targets,
            'metadata': {
                'video_id': sample['video_id'],
                'duration': sample['duration'],
                'category': sample['category'],
                'viral_patterns': sample['viral_patterns']
            }
        }


class ViralVideoLoss(nn.Module):
    """Multi-task loss function for viral video training"""
    
    def __init__(self, config: ViralVideoTrainingArguments):
        super().__init__()
        self.config = config
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.contrastive_loss = ContrastiveLoss()
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss"""
        
        losses = {}
        total_loss = 0.0
        
        # Video understanding loss
        if 'viral_potential' in predictions and 'viral_score' in targets:
            viral_loss = self.mse_loss(predictions['viral_potential'], targets['viral_score'])
            losses['viral_loss'] = viral_loss
            total_loss += self.config.viral_prediction_weight * viral_loss
        
        # Highlight detection loss
        if 'highlights' in predictions and 'highlight_labels' in targets:
            highlight_loss = self.bce_loss(predictions['highlights'], targets['highlight_labels'])
            losses['highlight_loss'] = highlight_loss
            total_loss += self.config.highlight_detection_weight * highlight_loss
        
        # Scene change detection loss
        if 'scene_changes' in predictions and 'scene_changes' in targets:
            scene_loss = self.bce_loss(predictions['scene_changes'], targets['scene_changes'])
            losses['scene_loss'] = scene_loss
            total_loss += self.config.video_understanding_weight * scene_loss
        
        # Caption generation loss
        if 'caption_logits' in predictions and 'caption_tokens' in targets:
            caption_loss = self.ce_loss(
                predictions['caption_logits'].view(-1, predictions['caption_logits'].size(-1)),
                targets['caption_tokens'].view(-1)
            )
            losses['caption_loss'] = caption_loss
            total_loss += self.config.caption_generation_weight * caption_loss
        
        # Engagement prediction loss
        if 'engagement_scores' in predictions and 'engagement_rate' in targets:
            engagement_loss = self.mse_loss(predictions['engagement_scores'], targets['engagement_rate'])
            losses['engagement_loss'] = engagement_loss
            total_loss += self.config.engagement_prediction_weight * engagement_loss
        
        # Platform optimization loss
        if 'platform_predictions' in predictions and 'platform_targets' in targets:
            platform_loss = self.mse_loss(predictions['platform_predictions'], targets['platform_targets'])
            losses['platform_loss'] = platform_loss
            total_loss += 0.2 * platform_loss
        
        # Contrastive loss for better representations
        if self.config.use_contrastive_learning and 'video_features' in predictions:
            contrastive_loss = self.contrastive_loss(
                predictions['video_features'],
                targets.get('positive_pairs', None),
                targets.get('negative_pairs', None)
            )
            losses['contrastive_loss'] = contrastive_loss
            total_loss += self.config.contrastive_loss_weight * contrastive_loss
        
        losses['total_loss'] = total_loss
        return losses


class ContrastiveLoss(nn.Module):
    """Contrastive loss for learning better video representations"""
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        features: torch.Tensor,
        positive_pairs: Optional[torch.Tensor] = None,
        negative_pairs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute contrastive loss"""
        
        if positive_pairs is None:
            # Create positive pairs from batch (simplified)
            batch_size = features.size(0)
            positive_pairs = torch.arange(batch_size, device=features.device)
        
        # Normalize features
        features = F.normalize(features, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create labels for positive pairs
        labels = torch.arange(features.size(0), device=features.device)
        
        # Compute contrastive loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


class CurriculumScheduler:
    """Curriculum learning scheduler for progressive training difficulty"""
    
    def __init__(self, total_epochs: int, warmup_epochs: int = 3):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
    
    def step(self, epoch: int):
        """Update curriculum difficulty"""
        self.current_epoch = epoch
    
    def get_difficulty(self) -> float:
        """Get current difficulty level (0.0 to 1.0)"""
        if self.current_epoch < self.warmup_epochs:
            return 0.3 + 0.4 * (self.current_epoch / self.warmup_epochs)
        else:
            remaining_epochs = self.total_epochs - self.warmup_epochs
            progress = (self.current_epoch - self.warmup_epochs) / remaining_epochs
            return 0.7 + 0.3 * progress
    
    def should_include_sample(self, sample_difficulty: float) -> bool:
        """Determine if sample should be included based on current difficulty"""
        current_difficulty = self.get_difficulty()
        return sample_difficulty <= current_difficulty


class ViralVideoEvaluator:
    """Comprehensive evaluator for viral video model"""
    
    def __init__(self, model: ViralVideoClipsModel, config: ViralVideoClipsConfig):
        self.model = model
        self.config = config
        
    def evaluate_viral_prediction(
        self,
        dataloader: DataLoader,
        device: torch.device
    ) -> Dict[str, float]:
        """Evaluate viral prediction accuracy"""
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating viral prediction"):
                # Move to device
                video_frames = batch['video_frames'].to(device)
                targets = batch['viral_score'].to(device)
                
                # Forward pass
                outputs = self.model.video_transformer(video_frames)
                predictions = outputs['viral_potential']
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # Calculate metrics
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        
        # Binary classification metrics (viral vs non-viral)
        viral_threshold = 0.7
        pred_binary = (predictions > viral_threshold).astype(int)
        target_binary = (targets > viral_threshold).astype(int)
        
        accuracy = accuracy_score(target_binary, pred_binary)
        precision, recall, f1, _ = precision_recall_fscore_support(
            target_binary, pred_binary, average='binary'
        )
        
        try:
            auc = roc_auc_score(target_binary, predictions)
        except ValueError:
            auc = 0.5  # If only one class present
        
        return {
            'viral_prediction_mse': mse,
            'viral_prediction_mae': mae,
            'viral_prediction_accuracy': accuracy,
            'viral_prediction_precision': precision,
            'viral_prediction_recall': recall,
            'viral_prediction_f1': f1,
            'viral_prediction_auc': auc
        }
    
    def evaluate_highlight_detection(
        self,
        dataloader: DataLoader,
        device: torch.device
    ) -> Dict[str, float]:
        """Evaluate highlight detection performance"""
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating highlight detection"):
                # Move to device
                video_frames = batch['video_frames'].to(device)
                audio_waveform = batch['audio_waveform'].to(device)
                targets = batch['highlight_labels'].to(device)
                
                # Forward pass
                video_outputs = self.model.video_transformer(video_frames)
                audio_outputs = self.model.audio_processor(audio_waveform)
                highlight_outputs = self.model.highlight_detector(
                    video_outputs['video_features'],
                    audio_outputs['audio_features']
                )
                
                predictions = highlight_outputs['highlight_scores']
                
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # Binary classification metrics
        threshold = 0.5
        pred_binary = (predictions > threshold).astype(int)
        target_binary = (targets > threshold).astype(int)
        
        accuracy = accuracy_score(target_binary, pred_binary)
        precision, recall, f1, _ = precision_recall_fscore_support(
            target_binary, pred_binary, average='binary'
        )
        
        try:
            auc = roc_auc_score(target_binary, predictions)
        except ValueError:
            auc = 0.5
        
        return {
            'highlight_detection_accuracy': accuracy,
            'highlight_detection_precision': precision,
            'highlight_detection_recall': recall,
            'highlight_detection_f1': f1,
            'highlight_detection_auc': auc
        }
    
    def evaluate_engagement_prediction(
        self,
        dataloader: DataLoader,
        device: torch.device
    ) -> Dict[str, float]:
        """Evaluate engagement prediction accuracy"""
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating engagement prediction"):
                # Move to device
                video_frames = batch['video_frames'].to(device)
                audio_waveform = batch['audio_waveform'].to(device)
                targets = batch['engagement_rate'].to(device)
                
                # Forward pass
                video_outputs = self.model.video_transformer(video_frames)
                audio_outputs = self.model.audio_processor(audio_waveform)
                highlight_outputs = self.model.highlight_detector(
                    video_outputs['video_features'],
                    audio_outputs['audio_features']
                )
                
                predictions = highlight_outputs['engagement_scores']
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # Regression metrics
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        
        # Correlation
        correlation = np.corrcoef(predictions, targets)[0, 1]
        
        return {
            'engagement_prediction_mse': mse,
            'engagement_prediction_mae': mae,
            'engagement_prediction_correlation': correlation
        }
    
    def generate_evaluation_report(
        self,
        eval_dataloader: DataLoader,
        device: torch.device,
        output_path: str
    ) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        
        logging.info("Starting comprehensive evaluation...")
        
        # Evaluate different components
        viral_metrics = self.evaluate_viral_prediction(eval_dataloader, device)
        highlight_metrics = self.evaluate_highlight_detection(eval_dataloader, device)
        engagement_metrics = self.evaluate_engagement_prediction(eval_dataloader, device)
        
        # Combine all metrics
        all_metrics = {
            **viral_metrics,
            **highlight_metrics,
            **engagement_metrics
        }
        
        # Calculate overall score
        overall_score = (
            viral_metrics['viral_prediction_f1'] * 0.4 +
            highlight_metrics['highlight_detection_f1'] * 0.3 +
            engagement_metrics['engagement_prediction_correlation'] * 0.3
        )
        
        # Create report
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'overall_score': overall_score,
            'viral_prediction_metrics': viral_metrics,
            'highlight_detection_metrics': highlight_metrics,
            'engagement_prediction_metrics': engagement_metrics,
            'model_config': {
                'model_size': self.config.model_size,
                'hidden_size': self.config.hidden_size,
                'num_layers': self.config.num_hidden_layers
            }
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logging.info(f"Evaluation report saved to {output_path}")
        logging.info(f"Overall score: {overall_score:.3f}")
        
        return report


class ViralVideoTrainer:
    """Advanced trainer for viral video clips model"""
    
    def __init__(
        self,
        model: ViralVideoClipsModel,
        args: ViralVideoTrainingArguments,
        train_dataset: ViralVideoDataset,
        eval_dataset: Optional[ViralVideoDataset] = None,
        accelerator: Optional[Accelerator] = None
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.accelerator = accelerator
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not accelerator:
            self.model.to(self.device)
        
        # Setup data loaders
        self.train_dataloader = self._create_dataloader(train_dataset, shuffle=True)
        self.eval_dataloader = self._create_dataloader(eval_dataset, shuffle=False) if eval_dataset else None
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Setup loss function
        self.loss_fn = ViralVideoLoss(args)
        
        # Setup curriculum learning
        self.curriculum_scheduler = CurriculumScheduler(
            args.num_train_epochs
        ) if args.use_curriculum_learning else None
        
        # Setup evaluator
        self.evaluator = ViralVideoEvaluator(model, model.config)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = -float('inf') if args.greater_is_better else float('inf')
        
        # Setup logging
        self._setup_logging()
        
        # Setup accelerator
        if accelerator:
            (
                self.model,
                self.optimizer,
                self.train_dataloader,
                self.eval_dataloader,
                self.scheduler
            ) = accelerator.prepare(
                self.model,
                self.optimizer,
                self.train_dataloader,
                self.eval_dataloader,
                self.scheduler
            )
    
    def _create_dataloader(self, dataset: Optional[ViralVideoDataset], shuffle: bool = False) -> Optional[DataLoader]:
        """Create data loader"""
        if dataset is None:
            return None
        
        return DataLoader(
            dataset,
            batch_size=self.args.per_device_train_batch_size if shuffle else self.args.per_device_eval_batch_size,
            shuffle=shuffle,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=shuffle
        )
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with different learning rates for different components"""
        
        # Group parameters by component
        video_params = list(self.model.video_transformer.parameters())
        audio_params = list(self.model.audio_processor.parameters())
        caption_params = list(self.model.caption_generator.parameters())
        effects_params = list(self.model.effects_engine.parameters())
        highlight_params = list(self.model.highlight_detector.parameters())
        
        param_groups = [
            {'params': video_params, 'lr': self.args.video_learning_rate},
            {'params': audio_params, 'lr': self.args.audio_learning_rate},
            {'params': caption_params, 'lr': self.args.caption_learning_rate},
            {'params': effects_params, 'lr': self.args.effects_learning_rate},
            {'params': highlight_params, 'lr': self.args.learning_rate}
        ]
        
        if self.args.optimizer_type == "adamw":
            return AdamW(
                param_groups,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer_type == "sgd":
            return SGD(
                param_groups,
                lr=self.args.learning_rate,
                momentum=0.9,
                weight_decay=self.args.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.args.optimizer_type}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        
        total_steps = len(self.train_dataloader) * self.args.num_train_epochs
        
        if self.args.scheduler_type == "cosine_with_restarts":
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=total_steps // 4,
                T_mult=2
            )
        elif self.args.scheduler_type == "linear":
            return get_scheduler(
                "linear",
                optimizer=self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=total_steps
            )
        else:
            return get_scheduler(
                self.args.scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=total_steps
            )
    
    def _setup_logging(self):
        """Setup logging and monitoring"""
        
        if self.args.report_to == "wandb":
            wandb.init(
                project=self.args.project_name,
                name=self.args.run_name,
                config=self.args.__dict__
            )
    
    def train(self):
        """Main training loop"""
        
        logging.info("Starting training...")
        logging.info(f"Total epochs: {self.args.num_train_epochs}")
        logging.info(f"Total steps: {len(self.train_dataloader) * self.args.num_train_epochs}")
        
        for epoch in range(self.args.num_train_epochs):
            self.epoch = epoch
            
            # Update curriculum difficulty
            if self.curriculum_scheduler:
                self.curriculum_scheduler.step(epoch)
            
            # Train one epoch
            train_metrics = self._train_epoch()
            
            # Evaluate
            eval_metrics = {}
            if self.eval_dataloader and (epoch + 1) % (self.args.eval_steps // len(self.train_dataloader)) == 0:
                eval_metrics = self._evaluate()
            
            # Log metrics
            self._log_metrics(train_metrics, eval_metrics, epoch)
            
            # Save checkpoint
            if (epoch + 1) % (self.args.save_steps // len(self.train_dataloader)) == 0:
                self._save_checkpoint(epoch, eval_metrics)
            
            # Check for early stopping
            if self._should_stop_early(eval_metrics):
                logging.info("Early stopping triggered")
                break
        
        # Final evaluation and save
        if self.eval_dataloader:
            final_metrics = self._evaluate()
            self._log_metrics({}, final_metrics, self.args.num_train_epochs)
        
        self._save_final_model()
        
        logging.info("Training completed!")
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train one epoch"""
        
        self.model.train()
        epoch_losses = defaultdict(list)
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch + 1}/{self.args.num_train_epochs}"
        )
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            if not self.accelerator:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            
            # Forward pass
            outputs = self._forward_pass(batch)
            
            # Compute loss
            losses = self.loss_fn(outputs, batch)
            loss = losses['total_loss']
            
            # Backward pass
            if self.accelerator:
                self.accelerator.backward(loss)
            else:
                loss.backward()
            
            # Gradient clipping
            if self.args.max_grad_norm > 0:
                if self.accelerator:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            
            # Optimizer step
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Log losses
            for loss_name, loss_value in losses.items():
                epoch_losses[loss_name].append(loss_value.item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log step metrics
            if self.global_step % self.args.logging_steps == 0:
                step_metrics = {f"train/{k}": v.item() for k, v in losses.items()}
                step_metrics["train/learning_rate"] = self.scheduler.get_last_lr()[0]
                step_metrics["train/global_step"] = self.global_step
                
                if self.args.report_to == "wandb":
                    wandb.log(step_metrics, step=self.global_step)
        
        # Calculate epoch averages
        epoch_metrics = {f"train/{k}": np.mean(v) for k, v in epoch_losses.items()}
        
        return epoch_metrics
    
    def _forward_pass(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass through the model"""
        
        video_frames = batch['video_frames']
        audio_waveform = batch['audio_waveform']
        
        # Video analysis
        video_outputs = self.model.video_transformer(video_frames)
        
        # Audio analysis
        audio_outputs = self.model.audio_processor(audio_waveform)
        
        # Highlight detection
        highlight_outputs = self.model.highlight_detector(
            video_outputs['video_features'],
            audio_outputs['audio_features']
        )
        
        # Caption generation (if caption tokens provided)
        caption_outputs = {}
        if 'caption_tokens' in batch:
            caption_outputs = self.model.caption_generator(
                video_outputs['video_features'],
                audio_outputs['audio_features'],
                batch['caption_tokens']
            )
        
        # Combine outputs
        outputs = {
            **video_outputs,
            **audio_outputs,
            **highlight_outputs,
            **caption_outputs
        }
        
        return outputs
    
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate the model"""
        
        logging.info("Running evaluation...")
        
        # Generate comprehensive evaluation report
        eval_report = self.evaluator.generate_evaluation_report(
            self.eval_dataloader,
            self.device,
            os.path.join(self.args.output_dir, f"eval_report_epoch_{self.epoch}.json")
        )
        
        # Extract key metrics
        eval_metrics = {
            f"eval/{k}": v for k, v in eval_report['viral_prediction_metrics'].items()
        }
        eval_metrics.update({
            f"eval/{k}": v for k, v in eval_report['highlight_detection_metrics'].items()
        })
        eval_metrics.update({
            f"eval/{k}": v for k, v in eval_report['engagement_prediction_metrics'].items()
        })
        eval_metrics["eval/overall_score"] = eval_report['overall_score']
        
        return eval_metrics
    
    def _log_metrics(
        self,
        train_metrics: Dict[str, float],
        eval_metrics: Dict[str, float],
        epoch: int
    ):
        """Log training and evaluation metrics"""
        
        all_metrics = {**train_metrics, **eval_metrics}
        all_metrics["epoch"] = epoch
        
        # Log to console
        logging.info(f"Epoch {epoch + 1} metrics:")
        for key, value in all_metrics.items():
            if isinstance(value, float):
                logging.info(f"  {key}: {value:.4f}")
        
        # Log to wandb
        if self.args.report_to == "wandb":
            wandb.log(all_metrics, step=self.global_step)
    
    def _save_checkpoint(self, epoch: int, eval_metrics: Dict[str, float]):
        """Save model checkpoint"""
        
        checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model state
        if self.accelerator:
            self.accelerator.save_model(self.model, checkpoint_dir)
        else:
            torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
        
        # Save optimizer and scheduler
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'eval_metrics': eval_metrics
        }, os.path.join(checkpoint_dir, "training_state.pt"))
        
        # Save configuration
        with open(os.path.join(checkpoint_dir, "training_args.json"), 'w') as f:
            json.dump(self.args.__dict__, f, indent=2)
        
        logging.info(f"Checkpoint saved to {checkpoint_dir}")
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to save space"""
        
        checkpoint_dirs = []
        for item in os.listdir(self.args.output_dir):
            if item.startswith("checkpoint-"):
                checkpoint_dirs.append((
                    int(item.split("-")[1]),
                    os.path.join(self.args.output_dir, item)
                ))
        
        # Sort by step number and keep only the latest ones
        checkpoint_dirs.sort(key=lambda x: x[0])
        
        while len(checkpoint_dirs) > self.args.save_total_limit:
            _, dir_to_remove = checkpoint_dirs.pop(0)
            shutil.rmtree(dir_to_remove)
            logging.info(f"Removed old checkpoint: {dir_to_remove}")
    
    def _should_stop_early(self, eval_metrics: Dict[str, float]) -> bool:
        """Check if training should stop early"""
        
        if not eval_metrics or self.args.metric_for_best_model not in eval_metrics:
            return False
        
        current_metric = eval_metrics[self.args.metric_for_best_model]
        
        if self.args.greater_is_better:
            is_better = current_metric > self.best_metric
        else:
            is_better = current_metric < self.best_metric
        
        if is_better:
            self.best_metric = current_metric
            return False
        
        # Simple early stopping (could be more sophisticated)
        return False
    
    def _save_final_model(self):
        """Save the final trained model"""
        
        final_model_dir = os.path.join(self.args.output_dir, "final_model")
        os.makedirs(final_model_dir, exist_ok=True)
        
        if self.accelerator:
            self.accelerator.save_model(self.model, final_model_dir)
        else:
            torch.save(self.model.state_dict(), os.path.join(final_model_dir, "model.pt"))
        
        # Save configuration
        with open(os.path.join(final_model_dir, "config.yaml"), 'w') as f:
            yaml.dump(self.model.config.__dict__, f)
        
        logging.info(f"Final model saved to {final_model_dir}")
    
    def save_model(self, output_dir: str):
        """Save model to specified directory"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.accelerator:
            self.accelerator.save_model(self.model, output_dir)
        else:
            torch.save(self.model.state_dict(), os.path.join(output_dir, "model.pt"))
        
        # Save configuration
        with open(os.path.join(output_dir, "config.yaml"), 'w') as f:
            yaml.dump(self.model.config.__dict__, f)
        
        logging.info(f"Model saved to {output_dir}")


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = ViralVideoClipsConfig()
    
    # Create model
    model = ViralVideoClipsModel(config)
    
    # Create datasets
    train_dataset = ViralVideoDataset(split="train", config=config, max_samples=1000)
    eval_dataset = ViralVideoDataset(split="eval", config=config, max_samples=200)
    
    # Create training arguments
    training_args = ViralVideoTrainingArguments(
        output_dir="./output/viral_video_clips",
        num_train_epochs=5,
        per_device_train_batch_size=2,
        learning_rate=3e-5,
        eval_steps=100,
        save_steps=200,
        logging_steps=50
    )
    
    # Create trainer
    trainer = ViralVideoTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    # Start training
    trainer.train()
    
    print("Training completed successfully!")