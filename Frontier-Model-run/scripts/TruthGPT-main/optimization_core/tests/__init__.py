"""
TruthGPT Optimization Core Test Suite
Comprehensive testing framework for optimization algorithms and techniques
"""

from typing import List
from pathlib import Path

__version__ = "1.0.0"
__author__ = "TruthGPT Team"

# Test directories
TEST_DIRS = ['unit', 'integration', 'performance', 'fixtures']

def discover_tests() -> List[str]:
    """Discover all test files in the test suite"""
    test_files = []
    base_dir = Path(__file__).parent
    
    for test_dir in ['unit', 'integration', 'performance']:
        dir_path = base_dir / test_dir
        if dir_path.exists():
            for test_file in dir_path.glob("test_*.py"):
                test_files.append(str(test_file))
    
    return test_files

__all__ = ['discover_tests', 'TEST_DIRS', '__version__', '__author__']
