"""
Transfer Trainer
================

Main orchestrator for transfer learning.
"""
import torch
import time
import logging
from typing import Dict, Any
from .config import TransferLearningConfig
from .enums import TransferStrategy
from .fine_tuning import FineTuner
from .feature_extraction import FeatureExtractor
from .distillation import KnowledgeDistiller
from .adaptation import DomainAdapter
from .multitask_adapter import MultiTaskAdapter

logger = logging.getLogger(__name__)

class TransferTrainer:
    """Main transfer learning trainer"""
    
    def __init__(self, config: TransferLearningConfig):
        self.config = config
        
        # Components
        self.fine_tuner = FineTuner(config)
        self.feature_extractor = FeatureExtractor(config)
        self.knowledge_distiller = KnowledgeDistiller(config)
        self.domain_adapter = DomainAdapter(config)
        self.multi_task_adapter = MultiTaskAdapter(config)
        
        # Transfer learning state
        self.transfer_history = []
        
        logger.info("✅ Transfer Learning Trainer initialized")
    
    def train_transfer_learning(self, source_data: torch.Tensor, source_labels: torch.Tensor,
                               target_data: torch.Tensor, target_labels: torch.Tensor = None) -> Dict[str, Any]:
        """Train transfer learning"""
        logger.info(f"🚀 Training transfer learning with strategy: {self.config.transfer_strategy.value}")
        
        transfer_results = {
            'start_time': time.time(),
            'config': self.config,
            'stages': {}
        }
        
        # Stage 1: Fine-tuning
        if self.config.transfer_strategy == TransferStrategy.FINE_TUNING:
            logger.info("🔧 Stage 1: Fine-tuning")
            
            # Load pretrained model
            model = self.fine_tuner.load_pretrained_model(self.config.source_model_path)
            
            # Fine-tune model
            fine_tune_result = self.fine_tuner.fine_tune(model, target_data, target_labels)
            
            transfer_results['stages']['fine_tuning'] = fine_tune_result
        
        # Stage 2: Feature Extraction
        elif self.config.transfer_strategy == TransferStrategy.FEATURE_EXTRACTION:
            logger.info("🔍 Stage 2: Feature Extraction")
            
            feature_extraction_result = self.feature_extractor.train_feature_extractor(
                None, target_data, target_labels  # Model is dummy here
            )
            
            transfer_results['stages']['feature_extraction'] = feature_extraction_result
            
        # Stage 3: Knowledge Distillation
        elif self.config.transfer_strategy == TransferStrategy.KNOWLEDGE_DISTILLATION:
            logger.info("🎓 Stage 3: Knowledge Distillation")
            
            distillation_result = self.knowledge_distiller.distill_knowledge(
                source_data, source_labels, target_data, target_labels
            )
            
            transfer_results['stages']['knowledge_distillation'] = distillation_result
            
        # Stage 4: Domain Adaptation
        elif self.config.transfer_strategy == TransferStrategy.DOMAIN_ADAPTATION:
            logger.info("🔄 Stage 4: Domain Adaptation")
            
            adaptation_result = self.domain_adapter.adapt_domain(
                source_data, source_labels, target_data
            )
            
            transfer_results['stages']['domain_adaptation'] = adaptation_result
        
        # Final evaluation
        transfer_results['end_time'] = time.time()
        transfer_results['total_duration'] = transfer_results['end_time'] - transfer_results['start_time']
        
        # Store results
        self.transfer_history.append(transfer_results)
        
        logger.info("✅ Transfer learning completed")
        return transfer_results
    
    def generate_transfer_report(self, results: Dict[str, Any]) -> str:
        """Generate transfer learning report"""
        report = []
        report.append("=" * 50)
        report.append("TRANSFER LEARNING REPORT")
        report.append("=" * 50)
        
        # Configuration
        report.append("\nTRANSFER LEARNING CONFIGURATION:")
        report.append("-" * 30)
        report.append(f"Transfer Strategy: {self.config.transfer_strategy.value}")
        
        # Results
        report.append("\nTRANSFER LEARNING RESULTS:")
        report.append("-" * 25)
        report.append(f"Total Duration: {results.get('total_duration', 0):.2f} seconds")
        
        return "\n".join(report)

