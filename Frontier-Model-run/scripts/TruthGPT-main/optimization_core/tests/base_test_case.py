"""
Base test case for optimization_core tests.

Provides a base class with common setup, teardown, and utilities.
"""
import unittest
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

from .utils.test_helpers import (
    create_mock_engine,
    create_mock_processor,
    create_test_config,
    create_temp_directory,
    cleanup_temp_directory,
)
from .utils.test_fixtures import (
    TestConfig,
    MockInferenceEngine,
    MockDataProcessor,
    TestDataGenerator,
)

logger = logging.getLogger(__name__)


class BaseOptimizationCoreTestCase(unittest.TestCase):
    """Base test case for optimization_core tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        
        # Create temporary directory
        self.temp_dir = create_temp_directory()
        
        # Create test config
        self.test_config = TestConfig()
        
        # Create test data generator
        self.data_generator = TestDataGenerator()
        
        # Setup logging
        logging.basicConfig(level=logging.DEBUG)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary directory
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            cleanup_temp_directory(self.temp_dir)
        
        super().tearDown()
    
    def create_mock_engine(self, **kwargs):
        """Create a mock inference engine."""
        return create_mock_engine(**kwargs)
    
    def create_mock_processor(self, **kwargs):
        """Create a mock data processor."""
        return create_mock_processor(**kwargs)
    
    def create_test_config(self, **kwargs):
        """Create a test configuration."""
        return create_test_config(**kwargs)
    
    def assert_engine_works(self, engine, **kwargs):
        """Assert that engine works correctly."""
        from .utils.test_assertions import assert_engine_works
        assert_engine_works(engine, **kwargs)
    
    def assert_processor_works(self, processor, **kwargs):
        """Assert that processor works correctly."""
        from .utils.test_assertions import assert_processor_works
        assert_processor_works(processor, **kwargs)
    
    def assert_config_valid(self, config, **kwargs):
        """Assert that config is valid."""
        from .utils.test_assertions import assert_config_valid
        assert_config_valid(config, **kwargs)
    
    def assert_error_handled(self, func, **kwargs):
        """Assert that error is handled correctly."""
        from .utils.test_assertions import assert_error_handled
        assert_error_handled(func, **kwargs)













