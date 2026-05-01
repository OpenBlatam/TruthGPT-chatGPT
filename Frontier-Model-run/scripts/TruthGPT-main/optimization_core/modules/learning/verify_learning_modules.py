"""
Verification Script for Learning Modules
========================================

Verifies that all learning modules can be imported and instantiated.
"""
import sys
import logging
from pathlib import Path
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from optimization_core.modules.learning import (
    EvolutionaryOptimizer,
    CausalInferenceSystem,
    ActiveLearningSystem,
    AdaptiveLearningSystem,
    AdversarialLearningSystem,
    BayesianOptimizer,
    ContinualLearner,
    EnsembleManager,
    FederatedServer,
    HyperparameterOptimizer,
    MetaLearner,
    MultitaskModel,
    NeuralArchitectureSearch,
    RLSystem,
    SelfSupervisedTrainer,
    TransferLearningManager,
    # Configs
    EvolutionaryConfig,
    CausalConfig,
    ActiveLearningConfig,
    AdaptiveLearningConfig,
    AdversarialConfig,
    BayesianConfig,
    ContinualConfig,
    EnsembleConfig,
    FederatedConfig,
    HPOConfig,
    MetaConfig,
    MultitaskConfig,
    NASConfig,
    RLConfig,
    SelfSupervisedConfig,
    TransferLearningConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_modules():
    logger.info("🔍 Verifying Learning Modules...")
    
    try:
        # 1. Evolutionary
        logger.info("   Checking Evolutionary...")
        evo_config = EvolutionaryConfig()
        evo = EvolutionaryOptimizer(evo_config)
        logger.info("   ✅ Evolutionary OK")
        
        # 2. Causal
        logger.info("   Checking Causal...")
        causal_config = CausalConfig()
        causal = CausalInferenceSystem(causal_config)
        logger.info("   ✅ Causal OK")
        
        # 3. Active
        logger.info("   Checking Active...")
        active_config = ActiveLearningConfig()
        active = ActiveLearningSystem(active_config)
        logger.info("   ✅ Active OK")
        
        # 4. Adaptive
        logger.info("   Checking Adaptive...")
        adaptive_config = AdaptiveLearningConfig()
        adaptive = AdaptiveLearningSystem(adaptive_config)
        logger.info("   ✅ Adaptive OK")

        # 5. Adversarial
        logger.info("   Checking Adversarial...")
        adv_config = AdversarialConfig()
        adv = AdversarialLearningSystem(adv_config)
        logger.info("   ✅ Adversarial OK")
        
        # 6. Bayesian
        logger.info("   Checking Bayesian...")
        bayesian_config = BayesianConfig()
        bayesian = BayesianOptimizer(bayesian_config)
        logger.info("   ✅ Bayesian OK")
        
        # 7. Continual
        logger.info("   Checking Continual...")
        continual_config = ContinualConfig()
        continual = ContinualLearner(continual_config)
        logger.info("   ✅ Continual OK")
        
        # 8. Ensemble
        # Note: EnsembleManager might require base learners or specific init
        logger.info("   Checking Ensemble...")
        ensemble_config = EnsembleConfig()
        ensemble = EnsembleManager(ensemble_config)
        logger.info("   ✅ Ensemble OK")
        
        # 9. Federated
        logger.info("   Checking Federated...")
        fed_config = FederatedConfig()
        fed = FederatedServer(None, fed_config)
        logger.info("   ✅ Federated OK")
        
        # 10. HPO
        logger.info("   Checking HPO...")
        hpo_config = HPOConfig()
        hpo = HyperparameterOptimizer(hpo_config)
        logger.info("   ✅ HPO OK")
        
        # 11. Meta
        logger.info("   Checking Meta...")
        meta_config = MetaConfig()
        meta = MetaLearner(nn.Linear(1, 1), meta_config)
        logger.info("   ✅ Meta OK")
        
        # 12. Multitask
        logger.info("   Checking Multitask...")
        multitask_config = MultitaskConfig()
        multitask = MultitaskModel(multitask_config)
        logger.info("   ✅ Multitask OK")
        
        # 13. NAS
        logger.info("   Checking NAS...")
        nas_config = NASConfig()
        nas = NeuralArchitectureSearch(nas_config)
        logger.info("   ✅ NAS OK")
        
        # 14. Reinforcement
        logger.info("   Checking Reinforcement...")
        rl_config = RLConfig()
        rl = RLSystem(rl_config)
        logger.info("   ✅ Reinforcement OK")
        
        # 15. Self-Supervised
        logger.info("   Checking Self-Supervised...")
        ssl_config = SelfSupervisedConfig()
        ssl = SelfSupervisedTrainer(ssl_config)
        logger.info("   ✅ Self-Supervised OK")
        
        # 16. Transfer
        logger.info("   Checking Transfer...")
        transfer_config = TransferLearningConfig()
        transfer = TransferLearningManager(transfer_config)
        logger.info("   ✅ Transfer OK")
        
        logger.info("✅ All learning modules verified successfully!")
        
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    verify_modules()

