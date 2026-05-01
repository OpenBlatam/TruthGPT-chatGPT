"""
Ensemble Learning System
========================

Main orchestrator for ensemble learning workflows.
"""
import time
import logging
import numpy as np
from typing import Dict, Any, List
from .config import EnsembleConfig, EnsembleStrategy
from .base import BaseModel
from .voting import VotingEnsemble
from .stacking import StackingEnsemble
from .bagging import BaggingEnsemble
from .boosting import BoostingEnsemble
from .dynamic import DynamicEnsemble

logger = logging.getLogger(__name__)

class EnsembleTrainer:
    """Main ensemble learning trainer"""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        
        # Components
        self.voting_ensemble = VotingEnsemble(config)
        self.stacking_ensemble = StackingEnsemble(config)
        self.bagging_ensemble = BaggingEnsemble(config)
        self.boosting_ensemble = BoostingEnsemble(config)
        self.dynamic_ensemble = DynamicEnsemble(config)
        
        # Ensemble learning state
        self.ensemble_history = []
        
        logger.info("✅ Ensemble Learning Trainer initialized")
    
    def train_ensemble_learning(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train ensemble learning"""
        logger.info(f"🚀 Training ensemble learning with strategy: {self.config.ensemble_strategy.value}")
        
        ensemble_results = {
            'start_time': time.time(),
            'config': self.config,
            'stages': {}
        }
        
        # Stage 1: Voting Ensemble
        if self.config.ensemble_strategy == EnsembleStrategy.VOTING_ENSEMBLE:
            logger.info("🗳️ Stage 1: Voting Ensemble")
            
            # Create and add models
            for i in range(self.config.num_models):
                model_type = self.config.model_types[i % len(self.config.model_types)]
                model = BaseModel(i, model_type, self.config)
                self.voting_ensemble.add_model(model)
            
            # Train voting ensemble
            voting_result = self.voting_ensemble.train_ensemble(X, y)
            
            ensemble_results['stages']['voting_ensemble'] = voting_result
        
        # Stage 2: Stacking Ensemble
        elif self.config.ensemble_strategy == EnsembleStrategy.STACKING_ENSEMBLE:
            logger.info("📚 Stage 2: Stacking Ensemble")
            
            # Create and add models
            for i in range(self.config.num_models):
                model_type = self.config.model_types[i % len(self.config.model_types)]
                model = BaseModel(i, model_type, self.config)
                self.stacking_ensemble.add_model(model)
            
            # Train stacking ensemble
            stacking_result = self.stacking_ensemble.train_ensemble(X, y)
            
            ensemble_results['stages']['stacking_ensemble'] = stacking_result
        
        # Stage 3: Bagging Ensemble
        elif self.config.ensemble_strategy == EnsembleStrategy.BAGGING_ENSEMBLE:
            logger.info("🎒 Stage 3: Bagging Ensemble")
            
            # Train bagging ensemble
            bagging_result = self.bagging_ensemble.train_ensemble(X, y)
            
            ensemble_results['stages']['bagging_ensemble'] = bagging_result
        
        # Stage 4: Boosting Ensemble
        elif self.config.ensemble_strategy == EnsembleStrategy.BOOSTING_ENSEMBLE:
            logger.info("🚀 Stage 4: Boosting Ensemble")
            
            # Train boosting ensemble
            boosting_result = self.boosting_ensemble.train_ensemble(X, y)
            
            ensemble_results['stages']['boosting_ensemble'] = boosting_result
        
        # Stage 5: Dynamic Ensemble
        elif self.config.ensemble_strategy == EnsembleStrategy.DYNAMIC_ENSEMBLE:
            logger.info("⚡ Stage 5: Dynamic Ensemble")
            
            # Create and add models
            for i in range(self.config.num_models):
                model_type = self.config.model_types[i % len(self.config.model_types)]
                model = BaseModel(i, model_type, self.config)
                self.dynamic_ensemble.add_model(model)
            
            # Train dynamic ensemble
            dynamic_result = self.dynamic_ensemble.train_ensemble(X, y)
            
            ensemble_results['stages']['dynamic_ensemble'] = dynamic_result
        
        # Final evaluation
        ensemble_results['end_time'] = time.time()
        ensemble_results['total_duration'] = ensemble_results['end_time'] - ensemble_results['start_time']
        
        # Store results
        self.ensemble_history.append(ensemble_results)
        
        logger.info("✅ Ensemble learning training completed")
        return ensemble_results
    
    def generate_ensemble_report(self, results: Dict[str, Any]) -> str:
        """Generate ensemble learning report"""
        report = []
        report.append("=" * 50)
        report.append("ENSEMBLE LEARNING REPORT")
        report.append("=" * 50)
        
        # Configuration
        report.append("\nENSEMBLE LEARNING CONFIGURATION:")
        report.append("-" * 33)
        report.append(f"Ensemble Strategy: {self.config.ensemble_strategy.value}")
        report.append(f"Voting Strategy: {self.config.voting_strategy.value}")
        report.append(f"Boosting Method: {self.config.boosting_method.value}")
        report.append(f"Number of Models: {self.config.num_models}")
        
        # Results
        report.append("\nENSEMBLE LEARNING RESULTS:")
        report.append("-" * 28)
        report.append(f"Total Duration: {results.get('total_duration', 0):.2f} seconds")
        
        return "\n".join(report)

