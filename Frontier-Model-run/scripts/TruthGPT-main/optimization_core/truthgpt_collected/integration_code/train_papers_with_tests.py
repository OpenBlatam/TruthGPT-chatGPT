#!/usr/bin/env python3
"""
Entrenamiento de Modelos con Validación por Tests
==================================================

Entrena cada paper individualmente usando los unit tests para validación.
Incluye early stopping basado en tests y métricas de calidad.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import sys
import json
import time
from typing import Dict, List, Any, Tuple
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add papers to path
papers_dir = Path(__file__).parent / 'papers'
sys.path.insert(0, str(papers_dir))

# Import TruthGPT
from truthgpt_optimization_core_integration import (
    TruthGPTOptimizationCore,
    TruthGPTOptimizationCoreConfig
)

# Import unit test utilities
import importlib.util

def load_module(module_path, module_name):
    """Load a module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SyntheticDataset(Dataset):
    """Dataset sintético para entrenamiento."""
    
    def __init__(self, vocab_size=1000, seq_len=32, num_samples=1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random input
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        # Generate random target (shifted by 1 for language modeling)
        target_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return input_ids, target_ids


class PaperTrainer:
    """Entrenador para papers individuales con validación por tests."""
    
    def __init__(self, paper_name: str, config: TruthGPTOptimizationCoreConfig, 
                 device: str = 'cpu', learning_rate: float = 1e-4):
        self.paper_name = paper_name
        self.config = config
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        
        # Create model
        self.core = TruthGPTOptimizationCore(config)
        self.model = self.core.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Training state
        self.train_losses = []
        self.test_scores = []
        self.best_test_score = 0.0
        self.best_model_state = None
        
        logger.info(f"Initialized trainer for {paper_name}")
    
    def run_unit_test(self) -> Dict[str, Any]:
        """Ejecuta unit test para validar el modelo."""
        self.model.eval()
        
        try:
            # Create test input
            batch_size = 2
            seq_len = 32
            input_ids = torch.randint(0, self.config.vocab_size, 
                                     (batch_size, seq_len), 
                                     device=self.device)
            
            # Forward pass
            with torch.no_grad():
                output = self.model(input_ids)
            
            # Get metrics
            metrics = self.core.get_all_metrics()
            
            # Calculate test score (combination of metrics)
            test_score = 0.0
            
            # Check if output is valid
            if 'logits' in output and output['logits'].shape == (batch_size, seq_len, self.config.vocab_size):
                test_score += 0.5  # Shape is correct
            
            # Check if metrics are available
            if metrics:
                test_score += 0.3  # Metrics available
            
            # Check for paper-specific metrics
            paper_metric_key = self.paper_name.lower().replace(' ', '_').replace('-', '_')
            if paper_metric_key in metrics:
                test_score += 0.2  # Paper-specific metrics available
            
            return {
                'score': test_score,
                'valid': True,
                'output_shape': output['logits'].shape if 'logits' in output else None,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Unit test failed: {e}")
            return {
                'score': 0.0,
                'valid': False,
                'error': str(e)
            }
    
    def train_step(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> float:
        """Un paso de entrenamiento."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        output = self.model(input_ids)
        logits = output['logits']
        
        # Calculate loss (cross entropy)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=-1
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Entrena una época."""
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            loss = self.train_step(input_ids, target_ids)
            total_loss += loss
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.info(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss:.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def train(self, num_epochs: int = 5, test_every: int = 1, 
              early_stopping_patience: int = 3) -> Dict[str, Any]:
        """Entrena el modelo con validación por tests."""
        logger.info(f"Starting training for {self.paper_name}")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Test every: {test_every} epochs")
        logger.info(f"  Early stopping patience: {early_stopping_patience}")
        
        # Create dataset
        dataset = SyntheticDataset(
            vocab_size=self.config.vocab_size,
            seq_len=32,
            num_samples=500
        )
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Initial test
        initial_test = self.run_unit_test()
        logger.info(f"Initial test score: {initial_test['score']:.4f}")
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*60}")
            
            # Train epoch
            train_loss = self.train_epoch(dataloader)
            self.train_losses.append(train_loss)
            logger.info(f"Train loss: {train_loss:.4f}")
            
            # Run test if needed
            if (epoch + 1) % test_every == 0:
                test_result = self.run_unit_test()
                test_score = test_result['score']
                self.test_scores.append(test_score)
                
                logger.info(f"Test score: {test_score:.4f}")
                logger.info(f"Test valid: {test_result['valid']}")
                
                # Check if best
                if test_score > self.best_test_score:
                    self.best_test_score = test_score
                    self.best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                    logger.info(f"✅ New best test score: {test_score:.4f}")
                else:
                    patience_counter += 1
                    logger.info(f"⏳ No improvement ({patience_counter}/{early_stopping_patience})")
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Loaded best model (score: {self.best_test_score:.4f})")
        
        # Final test
        final_test = self.run_unit_test()
        
        return {
            'paper_name': self.paper_name,
            'num_epochs_trained': epoch + 1,
            'initial_test_score': initial_test['score'],
            'final_test_score': final_test['score'],
            'best_test_score': self.best_test_score,
            'train_losses': self.train_losses,
            'test_scores': self.test_scores,
            'improvement': final_test['score'] - initial_test['score'],
            'final_metrics': final_test.get('metrics', {})
        }


def train_all_papers(num_epochs: int = 5, test_every: int = 1):
    """Entrena todos los papers."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Papers configuration
    papers_config = [
        ("Qwen3", {
            "enable_qwen3": True,
            "hidden_size": 768,
            "vocab_size": 1000,
        }),
        ("Absolute Zero", {
            "enable_absolute_zero": True,
            "hidden_size": 768,
            "vocab_size": 1000,
        }),
        ("Seed1.5-VL", {
            "enable_seed1_5_vl": True,
            "hidden_size": 768,
            "vocab_size": 1000,
        }),
        ("Mixture of Reasonings", {
            "enable_mixture_of_reasonings": True,
            "hidden_size": 768,
            "vocab_size": 1000,
        }),
        ("CRFT", {
            "enable_crft": True,
            "hidden_size": 768,
            "vocab_size": 1000,
        }),
        ("Meta-CoT", {
            "enable_meta_cot": True,
            "hidden_size": 768,
            "vocab_size": 1000,
        }),
        ("SFT vs RL", {
            "enable_sft_rl_generalization": True,
            "hidden_size": 768,
            "vocab_size": 1000,
        }),
        ("Learning Dynamics", {
            "enable_learning_dynamics": True,
            "hidden_size": 768,
            "vocab_size": 1000,
        }),
        ("Faster Cascades", {
            "enable_faster_cascades": True,
            "hidden_size": 768,
            "vocab_size": 1000,
        }),
        ("DeepSeek-V3", {
            "enable_deepseek_v3": True,
            "hidden_size": 768,
            "vocab_size": 1000,
        }),
    ]
    
    results = []
    
    for paper_name, paper_config in papers_config:
        logger.info(f"\n{'='*80}")
        logger.info(f"Training: {paper_name}")
        logger.info(f"{'='*80}")
        
        try:
            config = TruthGPTOptimizationCoreConfig(**paper_config)
            trainer = PaperTrainer(paper_name, config, device=device)
            
            result = trainer.train(
                num_epochs=num_epochs,
                test_every=test_every,
                early_stopping_patience=3
            )
            
            results.append(result)
            logger.info(f"✅ {paper_name} training completed")
            
        except Exception as e:
            logger.error(f"❌ {paper_name} training failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'paper_name': paper_name,
                'error': str(e),
                'success': False
            })
    
    return results


def generate_training_report(results: List[Dict[str, Any]]) -> str:
    """Genera reporte de entrenamiento."""
    report = []
    report.append("="*80)
    report.append("📊 TRAINING REPORT - Top 10 Papers 2025")
    report.append("="*80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Summary
    successful = [r for r in results if r.get('success', True) and 'error' not in r]
    failed = [r for r in results if 'error' in r or not r.get('success', True)]
    
    report.append("## 📈 SUMMARY")
    report.append("")
    report.append(f"- Total papers: {len(results)}")
    report.append(f"- Successful: {len(successful)}")
    report.append(f"- Failed: {len(failed)}")
    report.append("")
    
    # Successful trainings
    if successful:
        report.append("## ✅ SUCCESSFUL TRAININGS")
        report.append("")
        report.append("| Paper | Initial Score | Final Score | Best Score | Improvement | Epochs |")
        report.append("|-------|---------------|-------------|------------|-------------|--------|")
        
        for result in successful:
            report.append(
                f"| {result['paper_name']} | "
                f"{result.get('initial_test_score', 0):.4f} | "
                f"{result.get('final_test_score', 0):.4f} | "
                f"{result.get('best_test_score', 0):.4f} | "
                f"{result.get('improvement', 0):+.4f} | "
                f"{result.get('num_epochs_trained', 0)} |"
            )
        report.append("")
    
    # Failed trainings
    if failed:
        report.append("## ❌ FAILED TRAININGS")
        report.append("")
        for result in failed:
            report.append(f"### {result['paper_name']}")
            report.append(f"Error: {result.get('error', 'Unknown error')}")
            report.append("")
    
    # Best improvements
    if successful:
        report.append("## 🏆 BEST IMPROVEMENTS")
        report.append("")
        sorted_by_improvement = sorted(successful, key=lambda x: x.get('improvement', 0), reverse=True)
        for i, result in enumerate(sorted_by_improvement[:5], 1):
            report.append(f"{i}. **{result['paper_name']}**: {result.get('improvement', 0):+.4f} improvement")
        report.append("")
    
    # Training curves
    if successful:
        report.append("## 📉 TRAINING CURVES")
        report.append("")
        for result in successful:
            if 'train_losses' in result and result['train_losses']:
                losses = result['train_losses']
                report.append(f"### {result['paper_name']}")
                report.append(f"- Initial loss: {losses[0]:.4f}")
                report.append(f"- Final loss: {losses[-1]:.4f}")
                report.append(f"- Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
                report.append("")
    
    report.append("="*80)
    
    return "\n".join(report)


def main():
    """Función principal."""
    logger.info("🚀 Starting Training with Test Validation...")
    logger.info("="*80)
    
    # Train all papers
    results = train_all_papers(num_epochs=5, test_every=1)
    
    # Generate report
    report = generate_training_report(results)
    
    # Save results
    output_dir = Path(__file__).parent / "training_results"
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON
    json_file = output_dir / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"💾 Results saved to: {json_file}")
    
    # Save report
    report_file = output_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_file, 'w') as f:
        f.write(report)
    logger.info(f"📄 Report saved to: {report_file}")
    
    # Print report
    print("\n" + report)
    
    logger.info("\n✅ Training complete!")


if __name__ == "__main__":
    main()



