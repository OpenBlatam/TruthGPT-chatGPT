
import sys
import os
import logging
import unittest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class TestOptimizationCore(unittest.TestCase):
    def test_imports(self):
        """Test that we can import main components via optimization_core"""
        try:
            from optimization_core import (
                EvolutionaryOptimizer,
                CausalInferenceSystem
            )
            print("Successfully imported EvolutionaryOptimizer and CausalInferenceSystem")
        except ImportError as e:
            self.fail(f"Failed to import components: {e}")

if __name__ == '__main__':
    unittest.main()

