"""
TruthGPT Evaluation Utilities
Comprehensive evaluation utilities for TruthGPT models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import time
from pathlib import Path
import json
from enum import Enum
import math
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class TruthGPTEvaluationConfig:
    """TruthGPT evaluation configuration."""
    # Model configuration
    model_name: str = "truthgpt"
    model_size: str = "base"  # base, large, xl
    precision: str = "fp16"  # fp32, fp16, bf16
    device: str = "auto"  # auto, cpu, cuda, cuda:0, etc.
    
    # Evaluation configuration
    batch_size: int = 32
    max_sequence_length: int = 2048
    num_beams: int = 4
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    
    # Metrics configuration
    enable_perplexity: bool = True
    enable_bleu: bool = True
    enable_rouge: bool = True
    enable_accuracy: bool = True
    enable_f1: bool = True
    
    # Generation configuration
    max_new_tokens: int = 100
    do_sample: bool = True
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'model_size': self.model_size,
            'precision': self.precision,
            'device': self.device,
            'batch_size': self.batch_size,
            'max_sequence_length': self.max_sequence_length,
            'num_beams': self.num_beams,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'enable_perplexity': self.enable_perplexity,
            'enable_bleu': self.enable_bleu,
            'enable_rouge': self.enable_rouge,
            'enable_accuracy': self.enable_accuracy,
            'enable_f1': self.enable_f1,
            'max_new_tokens': self.max_new_tokens,
            'do_sample': self.do_sample,
            'repetition_penalty': self.repetition_penalty,
            'length_penalty': self.length_penalty
        }

class TruthGPTEvaluator:
    """TruthGPT model evaluator."""
    
    def __init__(self, config: TruthGPTEvaluationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Setup device
        self.device = self._setup_device()
        
        # Evaluation metrics
        self.evaluation_results = {}
        self.metrics_history = []
    
    def _setup_device(self):
        """Setup evaluation device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                self.logger.info("Using CPU device")
        else:
            device = torch.device(self.config.device)
            self.logger.info(f"Using specified device: {device}")
        
        return device
    
    def evaluate_model(self, model: nn.Module, test_loader, 
                      task_type: str = "language_modeling") -> Dict[str, Any]:
        """Evaluate TruthGPT model."""
        self.logger.info(f"📊 Starting TruthGPT evaluation for {task_type}")
        
        # Move model to device
        model = model.to(self.device)
        model.eval()
        
        # Apply precision
        if self.config.precision == "fp16":
            model = model.half()
        elif self.config.precision == "bf16":
            model = model.bfloat16()
        
        # Initialize metrics
        metrics = defaultdict(list)
        
        # Evaluate on test data
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    batch = [item.to(self.device) for item in batch]
                else:
                    batch = batch.to(self.device)
                
                # Apply precision
                if self.config.precision == "fp16":
                    batch = batch.half() if isinstance(batch, torch.Tensor) else batch
                elif self.config.precision == "bf16":
                    batch = batch.bfloat16() if isinstance(batch, torch.Tensor) else batch
                
                # Compute metrics based on task type
                if task_type == "language_modeling":
                    batch_metrics = self._evaluate_language_modeling(model, batch)
                elif task_type == "classification":
                    batch_metrics = self._evaluate_classification(model, batch)
                elif task_type == "generation":
                    batch_metrics = self._evaluate_generation(model, batch)
                else:
                    self.logger.warning(f"Unknown task type: {task_type}")
                    continue
                
                # Accumulate metrics
                for metric_name, metric_value in batch_metrics.items():
                    metrics[metric_name].append(metric_value)
                
                # Log progress
                if batch_idx % 100 == 0:
                    self.logger.info(f"Evaluated {batch_idx} batches")
        
        # Compute final metrics
        final_metrics = {}
        for metric_name, metric_values in metrics.items():
            if metric_values:
                final_metrics[metric_name] = {
                    'mean': np.mean(metric_values),
                    'std': np.std(metric_values),
                    'min': np.min(metric_values),
                    'max': np.max(metric_values)
                }
        
        self.evaluation_results = final_metrics
        self.metrics_history.append(final_metrics)
        
        self.logger.info("✅ TruthGPT evaluation completed")
        return final_metrics
    
    def _evaluate_language_modeling(self, model: nn.Module, batch) -> Dict[str, float]:
        """Evaluate language modeling task."""
        metrics = {}
        
        if isinstance(batch, (list, tuple)):
            input_ids, labels = batch
        else:
            input_ids = batch
            labels = input_ids.clone()
        
        # Forward pass
        outputs = model(input_ids)
        
        # Compute perplexity
        if self.config.enable_perplexity:
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction='mean'
            )
            perplexity = torch.exp(loss).item()
            metrics['perplexity'] = perplexity
        
        # Compute accuracy
        if self.config.enable_accuracy:
            predictions = torch.argmax(outputs, dim=-1)
            correct = (predictions == labels).float()
            accuracy = correct.mean().item()
            metrics['accuracy'] = accuracy
        
        return metrics
    
    def _evaluate_classification(self, model: nn.Module, batch) -> Dict[str, float]:
        """Evaluate classification task."""
        metrics = {}
        
        if isinstance(batch, (list, tuple)):
            input_ids, labels = batch
        else:
            input_ids = batch
            labels = torch.randint(0, 2, (input_ids.size(0),))
        
        # Forward pass
        outputs = model(input_ids)
        
        # Compute accuracy
        if self.config.enable_accuracy:
            predictions = torch.argmax(outputs, dim=-1)
            correct = (predictions == labels).float()
            accuracy = correct.mean().item()
            metrics['accuracy'] = accuracy
        
        # Compute F1 score
        if self.config.enable_f1:
            predictions = torch.argmax(outputs, dim=-1)
            f1_score = self._compute_f1_score(predictions, labels)
            metrics['f1_score'] = f1_score
        
        return metrics
    
    def _evaluate_generation(self, model: nn.Module, batch) -> Dict[str, float]:
        """Evaluate generation task."""
        metrics = {}
        
        if isinstance(batch, (list, tuple)):
            input_ids, target_ids = batch
        else:
            input_ids = batch
            target_ids = input_ids.clone()
        
        # Generate text
        generated_ids = self._generate_text(model, input_ids)
        
        # Compute BLEU score
        if self.config.enable_bleu:
            bleu_score = self._compute_bleu_score(generated_ids, target_ids)
            metrics['bleu_score'] = bleu_score
        
        # Compute ROUGE score
        if self.config.enable_rouge:
            rouge_score = self._compute_rouge_score(generated_ids, target_ids)
            metrics['rouge_score'] = rouge_score
        
        return metrics
    
    def _generate_text(self, model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate text using the model."""
        model.eval()
        
        with torch.no_grad():
            # Simple greedy generation
            generated_ids = input_ids.clone()
            
            for _ in range(self.config.max_new_tokens):
                # Get next token probabilities
                outputs = model(generated_ids)
                next_token_logits = outputs[:, -1, :]
                
                # Apply temperature
                if self.config.temperature != 1.0:
                    next_token_logits = next_token_logits / self.config.temperature
                
                # Sample next token
                if self.config.do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # Stop if we hit max length
                if generated_ids.size(1) >= self.config.max_sequence_length:
                    break
        
        return generated_ids
    
    def _compute_f1_score(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute F1 score."""
        # Convert to numpy for easier computation
        pred_np = predictions.cpu().numpy()
        label_np = labels.cpu().numpy()
        
        # Compute precision and recall
        true_positives = np.sum((pred_np == 1) & (label_np == 1))
        false_positives = np.sum((pred_np == 1) & (label_np == 0))
        false_negatives = np.sum((pred_np == 0) & (label_np == 1))
        
        if true_positives + false_positives == 0:
            precision = 0.0
        else:
            precision = true_positives / (true_positives + false_positives)
        
        if true_positives + false_negatives == 0:
            recall = 0.0
        else:
            recall = true_positives / (true_positives + false_negatives)
        
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        
        return f1_score
    
    def _compute_bleu_score(self, generated_ids: torch.Tensor, target_ids: torch.Tensor) -> float:
        """Compute BLEU score."""
        # This is a simplified BLEU computation
        # In practice, you'd use a proper BLEU implementation
        generated_tokens = generated_ids.cpu().numpy()
        target_tokens = target_ids.cpu().numpy()
        
        # Simple token overlap as proxy for BLEU
        overlap = 0
        total = 0
        
        for gen_seq, target_seq in zip(generated_tokens, target_tokens):
            gen_set = set(gen_seq)
            target_set = set(target_seq)
            overlap += len(gen_set.intersection(target_set))
            total += len(target_set)
        
        if total == 0:
            return 0.0
        
        return overlap / total
    
    def _compute_rouge_score(self, generated_ids: torch.Tensor, target_ids: torch.Tensor) -> float:
        """Compute ROUGE score."""
        # This is a simplified ROUGE computation
        # In practice, you'd use a proper ROUGE implementation
        generated_tokens = generated_ids.cpu().numpy()
        target_tokens = target_ids.cpu().numpy()
        
        # Simple token overlap as proxy for ROUGE
        overlap = 0
        total = 0
        
        for gen_seq, target_seq in zip(generated_tokens, target_tokens):
            gen_set = set(gen_seq)
            target_set = set(target_seq)
            overlap += len(gen_set.intersection(target_set))
            total += len(target_set)
        
        if total == 0:
            return 0.0
        
        return overlap / total
    
    def benchmark_model(self, model: nn.Module, input_shape: Tuple[int, ...], 
                       num_runs: int = 100) -> Dict[str, float]:
        """Benchmark model performance."""
        self.logger.info(f"⚡ Benchmarking TruthGPT model with {num_runs} runs")
        
        model = model.to(self.device)
        model.eval()
        
        # Apply precision
        if self.config.precision == "fp16":
            model = model.half()
        elif self.config.precision == "bf16":
            model = model.bfloat16()
        
        # Create dummy input
        dummy_input = torch.randn(self.config.batch_size, *input_shape).to(self.device)
        
        if self.config.precision == "fp16":
            dummy_input = dummy_input.half()
        elif self.config.precision == "bf16":
            dummy_input = dummy_input.bfloat16()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Benchmark
        times = []
        memory_usage = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                # Measure time
                start_time = time.time()
                start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
                _ = model(dummy_input)
                
                end_time = time.time()
                end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
                times.append(end_time - start_time)
                memory_usage.append(end_memory - start_memory)
        
        # Compute statistics
        benchmark_results = {
            'avg_inference_time': np.mean(times),
            'std_inference_time': np.std(times),
            'min_inference_time': np.min(times),
            'max_inference_time': np.max(times),
            'avg_memory_usage': np.mean(memory_usage) / (1024 * 1024),  # MB
            'max_memory_usage': np.max(memory_usage) / (1024 * 1024),  # MB
            'throughput': self.config.batch_size / np.mean(times)
        }
        
        self.logger.info(f"📊 Benchmark results: {benchmark_results}")
        return benchmark_results
    
    def get_evaluation_results(self) -> Dict[str, Any]:
        """Get evaluation results."""
        return self.evaluation_results
    
    def save_evaluation_results(self, filepath: str):
        """Save evaluation results."""
        results_data = {
            'config': self.config.to_dict(),
            'results': self.evaluation_results,
            'history': self.metrics_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        self.logger.info(f"📊 Evaluation results saved to {filepath}")

class TruthGPTComparison:
    """TruthGPT model comparison utilities."""
    
    def __init__(self, config: TruthGPTEvaluationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.comparison_results = {}
    
    def compare_models(self, models: Dict[str, nn.Module], test_loader, 
                      task_type: str = "language_modeling") -> Dict[str, Any]:
        """Compare multiple TruthGPT models."""
        self.logger.info(f"🔄 Comparing {len(models)} TruthGPT models")
        
        comparison_results = {}
        
        for model_name, model in models.items():
            self.logger.info(f"Evaluating {model_name}")
            
            # Create evaluator
            evaluator = TruthGPTEvaluator(self.config)
            
            # Evaluate model
            results = evaluator.evaluate_model(model, test_loader, task_type)
            comparison_results[model_name] = results
        
        # Generate comparison summary
        summary = self._generate_comparison_summary(comparison_results)
        comparison_results['summary'] = summary
        
        self.comparison_results = comparison_results
        return comparison_results
    
    def _generate_comparison_summary(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comparison summary."""
        summary = {}
        
        # Get all metric names
        all_metrics = set()
        for model_results in results.values():
            all_metrics.update(model_results.keys())
        
        # Compare each metric
        for metric in all_metrics:
            metric_values = {}
            for model_name, model_results in results.items():
                if metric in model_results:
                    metric_values[model_name] = model_results[metric]['mean']
            
            if metric_values:
                best_model = max(metric_values, key=metric_values.get)
                worst_model = min(metric_values, key=metric_values.get)
                
                summary[metric] = {
                    'best_model': best_model,
                    'best_value': metric_values[best_model],
                    'worst_model': worst_model,
                    'worst_value': metric_values[worst_model],
                    'all_values': metric_values
                }
        
        return summary
    
    def get_comparison_results(self) -> Dict[str, Any]:
        """Get comparison results."""
        return self.comparison_results

# Factory functions
def create_truthgpt_evaluator(config: TruthGPTEvaluationConfig) -> TruthGPTEvaluator:
    """Create TruthGPT evaluator."""
    return TruthGPTEvaluator(config)

def create_truthgpt_comparison(config: TruthGPTEvaluationConfig) -> TruthGPTComparison:
    """Create TruthGPT comparison."""
    return TruthGPTComparison(config)

def quick_truthgpt_evaluation(model: nn.Module, test_loader, 
                            task_type: str = "language_modeling",
                            precision: str = "fp16") -> Dict[str, Any]:
    """Quick TruthGPT evaluation."""
    config = TruthGPTEvaluationConfig(
        precision=precision,
        enable_perplexity=True,
        enable_accuracy=True
    )
    
    evaluator = create_truthgpt_evaluator(config)
    return evaluator.evaluate_model(model, test_loader, task_type)

# Context managers
@contextmanager
def truthgpt_evaluation_context(model: nn.Module, test_loader, config: TruthGPTEvaluationConfig):
    """Context manager for TruthGPT evaluation."""
    evaluator = create_truthgpt_evaluator(config)
    try:
        yield evaluator
    finally:
        # Cleanup if needed
        pass

# Example usage
if __name__ == "__main__":
    # Example TruthGPT evaluation
    print("📊 TruthGPT Evaluation Demo")
    print("=" * 50)
    
    # Create a sample TruthGPT-style model
    class TruthGPTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(10000, 768)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(768, 12, 3072, dropout=0.1),
                num_layers=12
            )
            self.lm_head = nn.Linear(768, 10000)
        
        def forward(self, x):
            x = self.embedding(x)
            x = self.transformer(x)
            x = self.lm_head(x)
            return x
    
    # Create model and dummy data
    model = TruthGPTModel()
    dummy_data = torch.randint(0, 10000, (100, 512))
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(dummy_data),
        batch_size=32,
        shuffle=False
    )
    
    # Quick evaluation
    results = quick_truthgpt_evaluation(model, test_loader, "language_modeling")
    print(f"Evaluation results: {results}")
    
    print("✅ TruthGPT evaluation completed!")



