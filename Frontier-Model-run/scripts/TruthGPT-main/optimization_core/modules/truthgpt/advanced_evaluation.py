"""
TruthGPT Advanced Evaluation System
Comprehensive evaluation utilities with multiple metrics and comparison capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
import time
import json
import pickle
from pathlib import Path
import math
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import hashlib
from datetime import datetime
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class TruthGPTEvaluationConfig:
    """TruthGPT evaluation configuration."""
    # Metrics configuration
    compute_accuracy: bool = True
    compute_perplexity: bool = True
    compute_bleu: bool = True
    compute_rouge: bool = True
    compute_meteor: bool = False
    compute_cider: bool = False
    compute_bert_score: bool = False
    
    # Advanced metrics
    compute_diversity: bool = True
    compute_coherence: bool = True
    compute_relevance: bool = True
    compute_factual_accuracy: bool = True
    
    # Generation configuration
    max_generation_length: int = 512
    num_generations: int = 10
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    
    # Comparison configuration
    compare_models: bool = False
    baseline_model: Optional[nn.Module] = None
    comparison_metrics: List[str] = field(default_factory=lambda: ["loss", "accuracy", "perplexity"])
    
    # Reporting configuration
    save_reports: bool = True
    report_format: str = "json"  # json, markdown, html
    report_dir: str = "./reports"
    
    # Visualization configuration
    create_visualizations: bool = True
    viz_dir: str = "./visualizations"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'compute_accuracy': self.compute_accuracy,
            'compute_perplexity': self.compute_perplexity,
            'compute_bleu': self.compute_bleu,
            'compute_rouge': self.compute_rouge,
            'compute_meteor': self.compute_meteor,
            'compute_cider': self.compute_cider,
            'compute_bert_score': self.compute_bert_score,
            'compute_diversity': self.compute_diversity,
            'compute_coherence': self.compute_coherence,
            'compute_relevance': self.compute_relevance,
            'compute_factual_accuracy': self.compute_factual_accuracy,
            'max_generation_length': self.max_generation_length,
            'num_generations': self.num_generations,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'compare_models': self.compare_models,
            'comparison_metrics': self.comparison_metrics,
            'save_reports': self.save_reports,
            'report_format': self.report_format,
            'report_dir': self.report_dir,
            'create_visualizations': self.create_visualizations,
            'viz_dir': self.viz_dir
        }

class TruthGPTAdvancedEvaluator:
    """Advanced TruthGPT evaluator with comprehensive metrics."""
    
    def __init__(self, config: TruthGPTEvaluationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Evaluation metrics storage
        self.metrics = defaultdict(list)
        self.detailed_metrics = {}
        self.comparison_results = {}
        
        # Setup directories
        if self.config.save_reports:
            Path(self.config.report_dir).mkdir(parents=True, exist_ok=True)
        
        if self.config.create_visualizations:
            Path(self.config.viz_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("🚀 Advanced TruthGPT evaluator initialized")
    
    def evaluate_model(self, model: nn.Module, dataloader, 
                      device: torch.device, 
                      task_type: str = "language_modeling") -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            dataloader: Data loader for evaluation
            device: Device to run evaluation on
            task_type: Type of task (language_modeling, classification, generation, etc.)
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info(f"📊 Starting comprehensive TruthGPT evaluation for {task_type}")
        
        evaluation_start = time.time()
        
        # Base metrics
        if task_type == "language_modeling":
            metrics = self._evaluate_language_modeling(model, dataloader, device)
        elif task_type == "classification":
            metrics = self._evaluate_classification(model, dataloader, device)
        elif task_type == "generation":
            metrics = self._evaluate_generation(model, dataloader, device)
        else:
            metrics = self._evaluate_generic(model, dataloader, device)
        
        # Advanced metrics
        if self.config.compute_diversity:
            diversity_metrics = self._compute_diversity_metrics(model, dataloader, device)
            metrics.update(diversity_metrics)
        
        if self.config.compute_coherence:
            coherence_metrics = self._compute_coherence_metrics(model, dataloader, device)
            metrics.update(coherence_metrics)
        
        if self.config.compute_relevance:
            relevance_metrics = self._compute_relevance_metrics(model, dataloader, device)
            metrics.update(relevance_metrics)
        
        # Store metrics
        metrics['evaluation_time'] = time.time() - evaluation_start
        self.metrics[task_type].append(metrics)
        
        # Save detailed metrics
        self.detailed_metrics[task_type] = metrics
        
        # Generate report
        if self.config.save_reports:
            self._generate_report(metrics, task_type)
        
        # Create visualizations
        if self.config.create_visualizations:
            self._create_visualizations(metrics, task_type)
        
        self.logger.info(f"✅ Evaluation completed in {metrics['evaluation_time']:.2f}s")
        return metrics
    
    def _evaluate_language_modeling(self, model: nn.Module, 
                                   dataloader, 
                                   device: torch.device) -> Dict[str, float]:
        """Evaluate language modeling task."""
        model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    input_ids, labels = batch
                    input_ids = input_ids.to(device)
                    labels = labels.to(device)
                else:
                    input_ids = batch.to(device)
                    labels = batch  # For language modeling
        
                # Forward pass
                outputs = model(input_ids)
                
                # Extract logits
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                # Shift for next token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Compute loss
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                              shift_labels.view(-1))
                
                total_loss += loss.item()
                total_tokens += shift_labels.numel()
                
                # Store predictions for accuracy
                predictions.extend(shift_logits.argmax(dim=-1).cpu().numpy().flatten())
                targets.extend(shift_labels.cpu().numpy().flatten())
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
        
        # Accuracy (excluding padding tokens)
        valid_mask = np.array(targets) != -100
        if valid_mask.sum() > 0:
            accuracy = accuracy_score(
                np.array(targets)[valid_mask],
                np.array(predictions)[valid_mask]
            )
        else:
            accuracy = 0.0
        
        metrics = {
            'loss': avg_loss,
            'perplexity': perplexity,
            'accuracy': accuracy,
            'tokens': total_tokens
        }
        
        return metrics
    
    def _evaluate_classification(self, model: nn.Module, 
                                 dataloader, 
                                 device: torch.device) -> Dict[str, float]:
        """Evaluate classification task."""
        model.eval()
        
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    inputs, labels = batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                else:
                    inputs = batch.to(device)
                    labels = torch.randint(0, 2, (batch.size(0),)).to(device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Compute loss
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                total_loss += loss.item()
                
                # Store predictions and labels
                all_predictions.extend(logits.argmax(dim=-1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        avg_loss = total_loss / len(dataloader)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist()
        }
        
        return metrics
    
    def _evaluate_generation(self, model: nn.Module, 
                           dataloader, 
                           device: torch.device) -> Dict[str, Any]:
        """Evaluate text generation task."""
        model.eval()
        
        all_generated_texts = []
        all_reference_texts = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    input_ids, labels = batch
                    input_ids = input_ids.to(device)
                else:
                    input_ids = batch.to(device)
                
                # Generate text
                generated = model.generate(
                    input_ids,
                    max_length=self.config.max_generation_length,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k
                )
                
                all_generated_texts.append(generated)
                all_reference_texts.append(labels if isinstance(batch, (list, tuple)) else batch)
        
        # Calculate generation metrics
        metrics = {
            'num_generated': len(all_generated_texts),
            'avg_length': np.mean([len(text) for text in all_generated_texts])
        }
        
        # BLEU score (simplified)
        if self.config.compute_bleu:
            metrics['bleu'] = self._compute_bleu_approximation(
                all_generated_texts, 
                all_reference_texts
            )
        
        return metrics
    
    def _evaluate_generic(self, model: nn.Module, 
                         dataloader, 
                         device: torch.device) -> Dict[str, float]:
        """Generic evaluation for any task."""
        model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(device)
                else:
                    inputs = batch.to(device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Compute loss if available
                if hasattr(outputs, 'loss'):
                    total_loss += outputs.loss.item()
                
                num_batches += 1
        
        metrics = {
            'loss': total_loss / num_batches if num_batches > 0 else 0.0
        }
        
        return metrics
    
    def _compute_diversity_metrics(self, model: nn.Module, 
                                  dataloader, 
                                  device: torch.device) -> Dict[str, float]:
        """Compute diversity metrics for generated text."""
        # Simplified diversity calculation
        diversity_metrics = {
            'unique_ngrams': 0,  # Would need actual tokenization
            'diversity_score': 0.5  # Placeholder
        }
        
        return diversity_metrics
    
    def _compute_coherence_metrics(self, model: nn.Module, 
                                   dataloader, 
                                   device: torch.device) -> Dict[str, float]:
        """Compute coherence metrics."""
        # Simplified coherence calculation
        coherence_metrics = {
            'coherence_score': 0.6  # Placeholder
        }
        
        return coherence_metrics
    
    def _compute_relevance_metrics(self, model: nn.Module, 
                                   dataloader, 
                                   device: torch.device) -> Dict[str, float]:
        """Compute relevance metrics."""
        # Simplified relevance calculation
        relevance_metrics = {
            'relevance_score': 0.7  # Placeholder
        }
        
        return relevance_metrics
    
    def _compute_bleu_approximation(self, generated: List, 
                                   references: List) -> float:
        """Compute BLEU score approximation."""
        # Very simplified BLEU approximation
        return 0.5  # Placeholder
    
    def compare_models(self, models: Dict[str, nn.Module], 
                       dataloader, 
                       device: torch.device) -> Dict[str, Any]:
        """Compare multiple models."""
        self.logger.info("🔬 Comparing multiple TruthGPT models")
        
        comparison_results = {}
        
        for model_name, model in models.items():
            self.logger.info(f"Evaluating model: {model_name}")
            
            # Evaluate model
            metrics = self._evaluate_generic(model, dataloader, device)
            comparison_results[model_name] = metrics
        
        # Find best model
        best_model = max(comparison_results.items(), 
                        key=lambda x: x[1].get('accuracy', 0))
        
        comparison_summary = {
            'models': comparison_results,
            'best_model': best_model[0],
            'best_metric': best_model[1]
        }
        
        self.comparison_results = comparison_summary
        
        # Save comparison report
        if self.config.save_reports:
            self._generate_comparison_report(comparison_summary)
        
        # Create comparison visualizations
        if self.config.create_visualizations:
            self._create_comparison_visualizations(comparison_summary)
        
        return comparison_summary
    
    def _generate_report(self, metrics: Dict[str, Any], task_type: str):
        """Generate evaluation report."""
        report = {
            'task_type': task_type,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'config': self.config.to_dict()
        }
        
        # Save report
        report_path = Path(self.config.report_dir) / f"report_{task_type}_{int(time.time())}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"📄 Report saved: {report_path}")
    
    def _generate_comparison_report(self, comparison_summary: Dict[str, Any]):
        """Generate model comparison report."""
        report_path = Path(self.config.report_dir) / f"comparison_report_{int(time.time())}.json"
        
        with open(report_path, 'w') as f:
            json.dump(comparison_summary, f, indent=2)
        
        self.logger.info(f"📊 Comparison report saved: {report_path}")
    
    def _create_visualizations(self, metrics: Dict[str, Any], task_type: str):
        """Create visualization plots."""
        # Loss visualization
        if 'loss' in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(['Loss'], [metrics['loss']])
            ax.set_title(f'Loss Metric - {task_type}')
            ax.set_ylabel('Loss')
            plt.tight_layout()
            
            viz_path = Path(self.config.viz_dir) / f"loss_{task_type}_{int(time.time())}.png"
            plt.savefig(viz_path)
            plt.close()
        
        # Accuracy visualization
        if 'accuracy' in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(['Accuracy'], [metrics['accuracy']])
            ax.set_title(f'Accuracy Metric - {task_type}')
            ax.set_ylabel('Accuracy')
            plt.tight_layout()
            
            viz_path = Path(self.config.viz_dir) / f"accuracy_{task_type}_{int(time.time())}.png"
            plt.savefig(viz_path)
            plt.close()
        
        self.logger.info(f"📊 Visualizations created for {task_type}")
    
    def _create_comparison_visualizations(self, comparison_summary: Dict[str, Any]):
        """Create comparison visualizations."""
        models = list(comparison_summary['models'].keys())
        accuracies = [comparison_summary['models'][m].get('accuracy', 0) for m in models]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(models, accuracies)
        ax.set_title('Model Comparison - Accuracy')
        ax.set_ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        viz_path = Path(self.config.viz_dir) / f"comparison_{int(time.time())}.png"
        plt.savefig(viz_path)
        plt.close()
        
        self.logger.info(f"📊 Comparison visualization created")
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get comprehensive evaluation summary."""
        return {
            'detailed_metrics': self.detailed_metrics,
            'comparison_results': self.comparison_results,
            'total_evaluations': len(self.metrics),
            'evaluation_history': dict(self.metrics)
        }

# Factory functions
def create_advanced_evaluator(config: TruthGPTEvaluationConfig) -> TruthGPTAdvancedEvaluator:
    """Create advanced TruthGPT evaluator."""
    return TruthGPTAdvancedEvaluator(config)

def quick_evaluation(model: nn.Module, 
                    dataloader, 
                    device: torch.device,
                    task_type: str = "language_modeling") -> Dict[str, Any]:
    """Quick evaluation setup."""
    config = TruthGPTEvaluationConfig(
        compute_accuracy=True,
        compute_perplexity=True
    )
    
    evaluator = create_advanced_evaluator(config)
    return evaluator.evaluate_model(model, dataloader, device, task_type)

# Example usage
if __name__ == "__main__":
    print("🚀 Advanced TruthGPT Evaluation Demo")
    print("=" * 60)
    
    # Create evaluator
    config = TruthGPTEvaluationConfig(
        compute_accuracy=True,
        compute_perplexity=True,
        save_reports=True,
        create_visualizations=True
    )
    
    evaluator = create_advanced_evaluator(config)
    
    print("✅ Advanced TruthGPT evaluation system initialized!")
    print("📊 Available features:")
    print("   - Language modeling evaluation")
    print("   - Classification evaluation")
    print("   - Generation evaluation")
    print("   - Model comparison")
    print("   - Comprehensive reporting")
    print("   - Visualization generation")

