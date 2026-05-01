"""
TruthGPT Evaluation Module
Advanced evaluation utilities for TruthGPT models following deep learning best practices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
from pathlib import Path
import json
import warnings

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field


class TruthGPTEvaluationConfig(BaseModel):
    """Configuration for TruthGPT evaluation."""
    # Evaluation settings
    eval_batch_size: int = 32
    max_eval_samples: int = 1000
    eval_steps: int = 100
    
    # Metrics configuration
    enable_perplexity: bool = True
    enable_accuracy: bool = True
    enable_bleu: bool = False
    enable_rouge: bool = False
    enable_meteor: bool = False
    
    # Advanced evaluation
    enable_beam_search: bool = False
    beam_size: int = 4
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    
    # Evaluation modes
    eval_mode: str = "validation"  # validation, test, inference
    enable_generation: bool = False
    max_generation_length: int = 100
    
    # Logging and monitoring
    log_eval_results: bool = True
    save_eval_results: bool = False
    eval_results_path: str = "./eval_results.json"

class TruthGPTMetrics(BaseModel):
    """Container for TruthGPT evaluation metrics."""
    # Basic metrics
    loss: float = 0.0
    perplexity: float = 0.0
    accuracy: float = 0.0
    
    # Generation metrics
    bleu_score: float = 0.0
    rouge_score: float = 0.0
    meteor_score: float = 0.0
    
    # Performance metrics
    eval_time: float = 0.0
    throughput: float = 0.0
    memory_used_mb: float = 0.0
    
    # Metadata
    eval_mode: str = "validation"
    num_samples: int = 0
    timestamp: float = Field(default_factory=time.time)

class TruthGPTEvaluator:
    """Advanced evaluator for TruthGPT models."""
    
    def __init__(self, config: TruthGPTEvaluationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Evaluation state
        self.evaluation_history = []
        self.best_metrics = {}
        
        self.logger.info("TruthGPT Evaluator initialized")
    
    def evaluate_model(self, model: nn.Module, eval_loader: torch.utils.data.DataLoader) -> TruthGPTMetrics:
        """Evaluate TruthGPT model."""
        self.logger.info(f"🔍 Evaluating TruthGPT model ({self.config.eval_mode})")
        
        start_time = time.time()
        model.eval()
        
        # Initialize metrics
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        num_samples = 0
        
        # Generation metrics
        generated_texts = []
        reference_texts = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_loader):
                # Prepare batch
                if isinstance(batch, (list, tuple)):
                    input_ids, labels = batch
                else:
                    input_ids = batch
                    labels = batch
                
                # Move to device
                input_ids = input_ids.to(next(model.parameters()).device)
                labels = labels.to(next(model.parameters()).device)
                
                # Forward pass
                outputs = model(input_ids)
                
                # Calculate loss
                loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(outputs, dim=-1)
                accuracy = (predictions == labels).float().mean()
                total_accuracy += accuracy.item()
                
                # Update counters
                num_batches += 1
                num_samples += input_ids.size(0)
                
                # Generation evaluation
                if self.config.enable_generation:
                    generated = self._generate_text(model, input_ids)
                    generated_texts.extend(generated)
                    reference_texts.extend([self._tokens_to_text(labels[i]) for i in range(labels.size(0))])
                
                # Limit evaluation samples
                if num_samples >= self.config.max_eval_samples:
                    break
        
        # Calculate final metrics
        eval_time = time.time() - start_time
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        throughput = num_samples / eval_time
        
        # Create metrics
        metrics = TruthGPTMetrics(
            loss=avg_loss,
            perplexity=perplexity,
            accuracy=avg_accuracy,
            eval_time=eval_time,
            throughput=throughput,
            eval_mode=self.config.eval_mode,
            num_samples=num_samples
        )
        
        # Calculate generation metrics
        if self.config.enable_generation and generated_texts:
            metrics = self._calculate_generation_metrics(metrics, generated_texts, reference_texts)
        
        # Update history
        self.evaluation_history.append(metrics)
        
        # Log results
        if self.config.log_eval_results:
            self._log_evaluation_results(metrics)
        
        # Save results
        if self.config.save_eval_results:
            self._save_evaluation_results(metrics)
        
        self.logger.info(f"✅ Evaluation completed - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}, Accuracy: {avg_accuracy:.4f}")
        
        return metrics
    
    def _generate_text(self, model: nn.Module, input_ids: torch.Tensor) -> List[str]:
        """Generate text using the model."""
        generated_texts = []
        
        for i in range(input_ids.size(0)):
            # Get input sequence
            input_seq = input_ids[i:i+1]
            
            # Generate text
            with torch.no_grad():
                if self.config.enable_beam_search:
                    generated = self._beam_search_generation(model, input_seq)
                else:
                    generated = self._greedy_generation(model, input_seq)
            
            generated_texts.append(generated)
        
        return generated_texts
    
    def _greedy_generation(self, model: nn.Module, input_ids: torch.Tensor) -> str:
        """Greedy text generation."""
        generated = input_ids.clone()
        
        for _ in range(self.config.max_generation_length):
            # Forward pass
            outputs = model(generated)
            next_token = torch.argmax(outputs[:, -1, :], dim=-1)
            
            # Append next token
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # Stop if EOS token
            if next_token.item() == 0:  # Assuming 0 is EOS token
                break
        
        return self._tokens_to_text(generated[0])
    
    def _beam_search_generation(self, model: nn.Module, input_ids: torch.Tensor) -> str:
        """Beam search text generation."""
        # Simplified beam search implementation
        beam_size = self.config.beam_size
        generated = input_ids.clone()
        
        for _ in range(self.config.max_generation_length):
            # Forward pass
            outputs = model(generated)
            logits = outputs[:, -1, :]
            
            # Get top-k tokens
            top_k_logits, top_k_tokens = torch.topk(logits, beam_size, dim=-1)
            
            # Select best token (simplified)
            next_token = top_k_tokens[0, 0].unsqueeze(0)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # Stop if EOS token
            if next_token.item() == 0:  # Assuming 0 is EOS token
                break
        
        return self._tokens_to_text(generated[0])
    
    def _tokens_to_text(self, tokens: torch.Tensor) -> str:
        """Convert tokens to text."""
        # Simplified token to text conversion
        return " ".join([str(token.item()) for token in tokens])
    
    def _calculate_generation_metrics(self, metrics: TruthGPTMetrics, 
                                   generated_texts: List[str], 
                                   reference_texts: List[str]) -> TruthGPTMetrics:
        """Calculate generation metrics."""
        if self.config.enable_bleu:
            metrics.bleu_score = self._calculate_bleu_score(generated_texts, reference_texts)
        
        if self.config.enable_rouge:
            metrics.rouge_score = self._calculate_rouge_score(generated_texts, reference_texts)
        
        if self.config.enable_meteor:
            metrics.meteor_score = self._calculate_meteor_score(generated_texts, reference_texts)
        
        return metrics
    
    def _calculate_bleu_score(self, generated_texts: List[str], reference_texts: List[str]) -> float:
        """Calculate BLEU score."""
        # Simplified BLEU calculation
        # In practice, you would use a proper BLEU implementation
        return 0.0
    
    def _calculate_rouge_score(self, generated_texts: List[str], reference_texts: List[str]) -> float:
        """Calculate ROUGE score."""
        # Simplified ROUGE calculation
        # In practice, you would use a proper ROUGE implementation
        return 0.0
    
    def _calculate_meteor_score(self, generated_texts: List[str], reference_texts: List[str]) -> float:
        """Calculate METEOR score."""
        # Simplified METEOR calculation
        # In practice, you would use a proper METEOR implementation
        return 0.0
    
    def _log_evaluation_results(self, metrics: TruthGPTMetrics) -> None:
        """Log evaluation results."""
        self.logger.info(f"📊 Evaluation Results ({metrics.eval_mode}):")
        self.logger.info(f"  Loss: {metrics.loss:.4f}")
        self.logger.info(f"  Perplexity: {metrics.perplexity:.4f}")
        self.logger.info(f"  Accuracy: {metrics.accuracy:.4f}")
        self.logger.info(f"  Eval Time: {metrics.eval_time:.2f}s")
        self.logger.info(f"  Throughput: {metrics.throughput:.2f} samples/s")
        
        if metrics.bleu_score > 0:
            self.logger.info(f"  BLEU Score: {metrics.bleu_score:.4f}")
        if metrics.rouge_score > 0:
            self.logger.info(f"  ROUGE Score: {metrics.rouge_score:.4f}")
        if metrics.meteor_score > 0:
            self.logger.info(f"  METEOR Score: {metrics.meteor_score:.4f}")
    
    def _save_evaluation_results(self, metrics: TruthGPTMetrics) -> None:
        """Save evaluation results."""
        results_path = Path(self.config.eval_results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(results_path, 'w') as f:
            json.dump(metrics.model_dump(), f, indent=2)
        
        self.logger.info(f"Evaluation results saved to {results_path}")
    
    def get_evaluation_history(self) -> List[TruthGPTMetrics]:
        """Get evaluation history."""
        return self.evaluation_history
    
    def get_best_metrics(self) -> Dict[str, Any]:
        """Get best metrics achieved."""
        if not self.evaluation_history:
            return {}
        
        # Find best metrics based on perplexity
        best_metrics = min(self.evaluation_history, key=lambda x: x.perplexity)
        return best_metrics.model_dump()
    
    def compare_models(self, model1: nn.Module, model2: nn.Module, 
                      eval_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """Compare two models."""
        self.logger.info("🔍 Comparing TruthGPT models")
        
        # Evaluate both models
        metrics1 = self.evaluate_model(model1, eval_loader)
        metrics2 = self.evaluate_model(model2, eval_loader)
        
        # Create comparison
        comparison = {
            'model1': metrics1.model_dump(),
            'model2': metrics2.model_dump(),
            'comparison': {
                'loss_diff': metrics1.loss - metrics2.loss,
                'perplexity_diff': metrics1.perplexity - metrics2.perplexity,
                'accuracy_diff': metrics1.accuracy - metrics2.accuracy,
                'better_model': 'model1' if metrics1.perplexity < metrics2.perplexity else 'model2'
            }
        }
        
        self.logger.info(f"Model comparison completed - Better model: {comparison['comparison']['better_model']}")
        
        return comparison

class TruthGPTMetricsCalculator:
    """Advanced metrics calculator for TruthGPT models."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def calculate_perplexity(self, loss: float) -> float:
        """Calculate perplexity from loss."""
        return torch.exp(torch.tensor(loss)).item()
    
    def calculate_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate accuracy."""
        return (predictions == targets).float().mean().item()
    
    def calculate_bleu_score(self, predictions: List[str], references: List[str]) -> float:
        """Calculate BLEU score."""
        # Simplified BLEU calculation
        # In practice, you would use a proper BLEU implementation
        return 0.0
    
    def calculate_rouge_score(self, predictions: List[str], references: List[str]) -> float:
        """Calculate ROUGE score."""
        # Simplified ROUGE calculation
        # In practice, you would use a proper ROUGE implementation
        return 0.0
    
    def calculate_meteor_score(self, predictions: List[str], references: List[str]) -> float:
        """Calculate METEOR score."""
        # Simplified METEOR calculation
        # In practice, you would use a proper METEOR implementation
        return 0.0

# Factory functions
def create_truthgpt_evaluator(config: TruthGPTEvaluationConfig) -> TruthGPTEvaluator:
    """Create TruthGPT evaluator."""
    return TruthGPTEvaluator(config)

def evaluate_truthgpt_model(model: nn.Module, eval_loader: torch.utils.data.DataLoader, 
                           config: TruthGPTEvaluationConfig) -> TruthGPTMetrics:
    """Quick TruthGPT model evaluation."""
    evaluator = create_truthgpt_evaluator(config)
    return evaluator.evaluate_model(model, eval_loader)

# Example usage
if __name__ == "__main__":
    # Example TruthGPT evaluation
    print("🚀 TruthGPT Evaluation Demo")
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
    
    # Create model
    model = TruthGPTModel()
    
    # Create dummy data
    dummy_data = [(torch.randint(0, 10000, (32, 512)), torch.randint(0, 10000, (32, 512))) for _ in range(10)]
    eval_loader = torch.utils.data.DataLoader(dummy_data, batch_size=32)
    
    # Create configuration
    config = TruthGPTEvaluationConfig(
        eval_batch_size=32,
        enable_perplexity=True,
        enable_accuracy=True,
        log_eval_results=True
    )
    
    # Create evaluator
    evaluator = create_truthgpt_evaluator(config)
    
    # Evaluate model
    metrics = evaluator.evaluate_model(model, eval_loader)
    
    print(f"Evaluation metrics: {metrics.model_dump()}")
    print("✅ TruthGPT evaluation completed!")



