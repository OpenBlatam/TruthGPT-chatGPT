import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Find the 'optimization_core' directory and add its PARENT to sys.path
# This allows 'import optimization_core' to work.
current_dir = os.path.dirname(os.path.abspath(__file__))
# current is .../optimization_core/modules/training
# we want .../TruthGPT-main (which contains optimization_core)
opt_core_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir))) 
# optimization_core/modules/training -> optimization_core/modules -> optimization_core -> TruthGPT-main
# Wait, current_dir is .../training
# dirname(current) is .../modules
# dirname(dirname(current)) is .../optimization_core
# dirname(dirname(dirname(current))) is .../TruthGPT-main
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, root_dir)

print(f"Added {root_dir} to sys.path")

try:
    from optimization_core.modules.training.config import TrainingConfig, TrainingStrategy
    from optimization_core.modules.training.trainer import AdvancedTrainer
    print("Imports successful via optimization_core!")
except ImportError as e:
    print(f"Import failed: {e}")
    # Try local fallback if optimization_core is not a package
    try:
        sys.path.append(current_dir)
        from config import TrainingConfig, TrainingStrategy
        from trainer import AdvancedTrainer
        print("Local imports successful!")
    except ImportError as e2:
         print(f"Local import failed: {e2}")
         sys.exit(1)

def test_training_refactor():
    print("Testing TrainingConfig instantiation...")
    config = TrainingConfig(
        epochs=1,
        batch_size=2,
        use_ema=True,
        strategy=TrainingStrategy.STANDARD
    )
    print("TrainingConfig instantiated.")

    print("Testing AdvancedTrainer instantiation...")
    model = nn.Linear(10, 2)
    dataset = TensorDataset(torch.randn(10, 10), torch.randint(0, 2, (10,)))
    dataloader = DataLoader(dataset, batch_size=2)
    
    trainer = AdvancedTrainer(config, model, dataloader)
    print("AdvancedTrainer instantiated.")
    
    # Check if components are initialized
    assert trainer.ema is not None, "EMA should be initialized"
    assert trainer.experiment_logger is not None, "ExperimentLogger should be initialized"
    
    print("Refactor verification PASSED.")

if __name__ == "__main__":
    test_training_refactor()

