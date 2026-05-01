
import sys
import os
import torch
import torch.nn as nn

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, root_dir)

print(f"Added {root_dir} to sys.path")

try:
    from optimization_core.modules.learning import (
        create_adversarial_learning_system,
        create_adversarial_config,
        AdversarialAttackType,
        GANType,
        DefenseStrategy
    )
    print("Imports successful!")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_adversarial_learning():
    print("Testing AdversarialLearningSystem...")
    config = create_adversarial_config(
        attack_type=AdversarialAttackType.PGD,
        gan_type=GANType.VANILLA_GAN
    )
    system = create_adversarial_learning_system(config)
    
    # Create dummy model
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    
    # Dummy data
    train_data = torch.randn(20, 10)
    train_labels = torch.randint(0, 2, (20,))
    test_data = torch.randn(10, 10)
    test_labels = torch.randint(0, 2, (10,))
    
    print("Running system...")
    results = system.run_adversarial_learning(model, train_data, train_labels, test_data, test_labels)
    
    print("Generating report...")
    report = system.generate_adversarial_report(results)
    print("Report generated.")
    
    if 'stages' in results:
        print("✅ Adversarial system successfully verified!")
    else:
        print("❌ Result structure invalid!")
        sys.exit(1)

if __name__ == "__main__":
    test_adversarial_learning()

