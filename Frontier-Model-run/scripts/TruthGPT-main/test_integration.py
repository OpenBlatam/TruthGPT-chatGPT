#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Frontier-Model-run'))

def test_integration():
    """Test that the native model integrates with training framework."""
    print("Testing integration with GRPO training framework...")
    
    try:
        from scripts.kf_grpo_train import main, KFGRPOScriptArguments
        from models.deepseek_v3 import create_deepseek_v3_model
        print("✓ Import test passed - native model integrates with training framework")
        return True
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)
