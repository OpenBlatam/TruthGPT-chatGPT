
import sys
import os
import importlib

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, root_dir)

print(f"sys.path[0]: {sys.path[0]}")

try:
    import optimization_core.modules.learning as learning
    print(f"Module imported: {learning}")
    print(f"Module __package__: {learning.__package__}")
    
    name = 'create_adversarial_learning_system'
    print(f"Attempting to get attribute '{name}'...")
    attr = getattr(learning, name)
    print(f"Successfully got attribute: {attr}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

