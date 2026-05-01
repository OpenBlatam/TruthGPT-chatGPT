"""
HPO System
==========

Main orchestrator for hyperparameter optimization workflows.
"""
import time
import logging
from typing import Dict, Any, Callable
from .config import HpoConfig, HpoAlgorithm
from .bayesian import BayesianOptimizer
from .evolutionary import EvolutionaryOptimizer
from .tpe import TPEOptimizer
from .cma_es import CMAESOptimizer
from .optuna import OptunaOptimizer
from .multi_objective import MultiObjectiveOptimizer

logger = logging.getLogger(__name__)

class HpoManager:
    """Main hyperparameter optimization manager"""
    
    def __init__(self, config: HpoConfig):
        self.config = config
        
        # Components
        self.bayesian_optimizer = BayesianOptimizer(config)
        self.evolutionary_optimizer = EvolutionaryOptimizer(config)
        self.tpe_optimizer = TPEOptimizer(config)
        self.cmaes_optimizer = CMAESOptimizer(config)
        self.optuna_optimizer = OptunaOptimizer(config)
        self.multi_objective_optimizer = MultiObjectiveOptimizer(config)
        
        # HPO state
        self.hpo_history = []
        
        logger.info("✅ HPO Manager initialized")
    
    def optimize_hyperparameters(self, objective_function: Callable, 
                               search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize hyperparameters"""
        logger.info(f"🚀 Optimizing hyperparameters using algorithm: {self.config.hpo_algorithm.value}")
        
        hpo_results = {
            'start_time': time.time(),
            'config': self.config,
            'stages': {}
        }
        
        # Stage 1: Bayesian Optimization
        if self.config.hpo_algorithm == HpoAlgorithm.BAYESIAN_OPTIMIZATION:
            logger.info("🔍 Stage 1: Bayesian Optimization")
            
            bayesian_result = self.bayesian_optimizer.optimize(objective_function, search_space)
            
            hpo_results['stages']['bayesian_optimization'] = bayesian_result
        
        # Stage 2: Evolutionary Algorithm
        elif self.config.hpo_algorithm == HpoAlgorithm.EVOLUTIONARY_ALGORITHM:
            logger.info("🧬 Stage 2: Evolutionary Algorithm")
            
            evolutionary_result = self.evolutionary_optimizer.optimize(objective_function, search_space)
            
            hpo_results['stages']['evolutionary_algorithm'] = evolutionary_result
        
        # Stage 3: TPE
        elif self.config.hpo_algorithm == HpoAlgorithm.TPE:
            logger.info("🌳 Stage 3: TPE")
            
            tpe_result = self.tpe_optimizer.optimize(objective_function, search_space)
            
            hpo_results['stages']['tpe'] = tpe_result
        
        # Stage 4: CMA-ES
        elif self.config.hpo_algorithm == HpoAlgorithm.CMA_ES:
            logger.info("🎯 Stage 4: CMA-ES")
            
            cmaes_result = self.cmaes_optimizer.optimize(objective_function, search_space)
            
            hpo_results['stages']['cma_es'] = cmaes_result
        
        # Stage 5: Optuna
        elif self.config.hpo_algorithm == HpoAlgorithm.OPTUNA:
            logger.info("🔬 Stage 5: Optuna")
            
            optuna_result = self.optuna_optimizer.optimize(objective_function, search_space)
            
            hpo_results['stages']['optuna'] = optuna_result
        
        # Stage 6: Multi-Objective Optimization
        elif self.config.enable_multi_objective:
            logger.info("🎯 Stage 6: Multi-Objective Optimization")
            
            multi_objective_result = self.multi_objective_optimizer.optimize(objective_function, search_space)
            
            hpo_results['stages']['multi_objective'] = multi_objective_result
        
        # Final evaluation
        hpo_results['end_time'] = time.time()
        hpo_results['total_duration'] = hpo_results['end_time'] - hpo_results['start_time']
        
        # Store results
        self.hpo_history.append(hpo_results)
        
        logger.info("✅ Hyperparameter optimization completed")
        return hpo_results
    
    def generate_hpo_report(self, results: Dict[str, Any]) -> str:
        """Generate HPO report"""
        report = []
        report.append("=" * 50)
        report.append("HYPERPARAMETER OPTIMIZATION REPORT")
        report.append("=" * 50)
        
        # Configuration
        report.append("\nHPO CONFIGURATION:")
        report.append("-" * 18)
        report.append(f"HPO Algorithm: {self.config.hpo_algorithm.value}")
        report.append(f"Sampler Type: {self.config.sampler_type.value}")
        report.append(f"Pruner Type: {self.config.pruner_type.value}")
        
        # Results
        report.append("\nHPO RESULTS:")
        report.append("-" * 13)
        report.append(f"Total Duration: {results.get('total_duration', 0):.2f} seconds")
        
        return "\n".join(report)

