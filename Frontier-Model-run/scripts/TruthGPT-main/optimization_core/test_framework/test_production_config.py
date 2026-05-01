"""
Test suite for Production Configuration System
Comprehensive tests for enterprise-grade configuration management
"""

import unittest
import tempfile
import os
import json
import yaml
import time
from unittest.mock import patch, MagicMock
from pathlib import Path

from production_config import (
    ProductionConfig, Environment, ConfigSource, ConfigValidationRule,
    create_production_config, load_config_from_file, create_environment_config,
    production_config_context, create_optimization_validation_rules,
    create_monitoring_validation_rules
)


class TestProductionConfig(unittest.TestCase):
    """Test cases for ProductionConfig class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.json")
        self.yaml_config_file = os.path.join(self.temp_dir, "test_config.yaml")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_default_config_initialization(self):
        """Test default configuration initialization."""
        config = ProductionConfig()
        
        # Test default values
        self.assertEqual(config.environment, Environment.DEVELOPMENT)
        self.assertTrue(config.enable_validation)
        self.assertFalse(config.enable_hot_reload)
        
        # Test default config data
        self.assertIn('optimization', config.config_data)
        self.assertIn('monitoring', config.config_data)
        self.assertIn('performance', config.config_data)
        self.assertIn('reliability', config.config_data)
        self.assertIn('caching', config.config_data)
    
    def test_environment_specific_config(self):
        """Test environment-specific configuration."""
        config = ProductionConfig(environment=Environment.PRODUCTION)
        self.assertEqual(config.environment, Environment.PRODUCTION)
        
        # Test environment-specific overrides
        env_config = config.get_environment_config()
        self.assertEqual(env_config['optimization']['level'], 'aggressive')
        self.assertEqual(env_config['monitoring']['log_level'], 'WARNING')
        self.assertEqual(env_config['reliability']['max_retry_attempts'], 5)
    
    def test_config_get_set_operations(self):
        """Test configuration get and set operations."""
        config = ProductionConfig()
        
        # Test setting and getting values
        config.set("test.value", 42)
        self.assertEqual(config.get("test.value"), 42)
        
        # Test nested values
        config.set("nested.deep.value", "test")
        self.assertEqual(config.get("nested.deep.value"), "test")
        
        # Test default values
        self.assertEqual(config.get("nonexistent.key", "default"), "default")
        self.assertIsNone(config.get("nonexistent.key"))
    
    def test_config_section_operations(self):
        """Test configuration section operations."""
        config = ProductionConfig()
        
        # Test getting entire section
        optimization_section = config.get_section("optimization")
        self.assertIsInstance(optimization_section, dict)
        self.assertIn("level", optimization_section)
        
        # Test updating section
        updates = {"new_setting": "value", "another_setting": 123}
        config.update_section("test_section", updates)
        
        self.assertEqual(config.get("test_section.new_setting"), "value")
        self.assertEqual(config.get("test_section.another_setting"), 123)
    
    def test_config_file_loading_json(self):
        """Test loading configuration from JSON file."""
        test_config = {
            "optimization": {
                "level": "aggressive",
                "enable_quantization": False
            },
            "monitoring": {
                "log_level": "DEBUG"
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(test_config, f)
        
        config = ProductionConfig(config_file=self.config_file)
        
        self.assertEqual(config.get("optimization.level"), "aggressive")
        self.assertFalse(config.get("optimization.enable_quantization"))
        self.assertEqual(config.get("monitoring.log_level"), "DEBUG")
    
    def test_config_file_loading_yaml(self):
        """Test loading configuration from YAML file."""
        test_config = {
            "optimization": {
                "level": "maximum",
                "enable_pruning": True
            }
        }
        
        with open(self.yaml_config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        config = ProductionConfig(config_file=self.yaml_config_file)
        
        self.assertEqual(config.get("optimization.level"), "maximum")
        self.assertTrue(config.get("optimization.enable_pruning"))
    
    def test_config_file_not_found(self):
        """Test handling of missing configuration file."""
        config = ProductionConfig(config_file="nonexistent.json")
        
        # Should fall back to default config
        self.assertIn('optimization', config.config_data)
        self.assertEqual(config.get("optimization.level"), "standard")
    
    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables."""
        with patch.dict(os.environ, {
            'OPTIMIZATION_LEVEL': 'aggressive',
            'OPTIMIZATION_MAX_MEMORY_GB': '32.0',
            'OPTIMIZATION_ENABLE_QUANTIZATION': 'true',
            'OPTIMIZATION_ENABLE_PRUNING': 'false'
        }):
            config = ProductionConfig()
            config.load_from_environment()
            
            self.assertEqual(config.get("optimization.level"), "aggressive")
            self.assertEqual(config.get("optimization.max_memory_gb"), 32.0)
            self.assertTrue(config.get("optimization.enable_quantization"))
            self.assertFalse(config.get("optimization.enable_pruning"))
    
    def test_environment_variable_parsing(self):
        """Test parsing of different environment variable types."""
        with patch.dict(os.environ, {
            'OPTIMIZATION_BOOLEAN_TRUE': 'true',
            'OPTIMIZATION_BOOLEAN_FALSE': 'false',
            'OPTIMIZATION_INTEGER': '42',
            'OPTIMIZATION_FLOAT': '3.14',
            'OPTIMIZATION_STRING': 'test_string',
            'OPTIMIZATION_LIST': 'item1,item2,item3'
        }):
            config = ProductionConfig()
            config.load_from_environment()
            
            self.assertTrue(config.get("optimization.boolean_true"))
            self.assertFalse(config.get("optimization.boolean_false"))
            self.assertEqual(config.get("optimization.integer"), 42)
            self.assertEqual(config.get("optimization.float"), 3.14)
            self.assertEqual(config.get("optimization.string"), "test_string")
            self.assertEqual(config.get("optimization.list"), ["item1", "item2", "item3"])
    
    def test_validation_rules(self):
        """Test configuration validation rules."""
        config = ProductionConfig()
        
        # Add validation rules
        rule1 = ConfigValidationRule(
            field_path="optimization.max_memory_gb",
            validator=lambda x: isinstance(x, (int, float)) and x > 0,
            error_message="max_memory_gb must be positive"
        )
        rule2 = ConfigValidationRule(
            field_path="optimization.level",
            validator=lambda x: x in ['minimal', 'standard', 'aggressive', 'maximum'],
            error_message="level must be valid"
        )
        
        config.add_validation_rule(rule1)
        config.add_validation_rule(rule2)
        
        # Test valid configuration
        config.set("optimization.max_memory_gb", 16.0)
        config.set("optimization.level", "standard")
        errors = config.validate_config()
        self.assertEqual(len(errors), 0)
        
        # Test invalid configuration
        config.set("optimization.max_memory_gb", -1.0)
        config.set("optimization.level", "invalid")
        errors = config.validate_config()
        self.assertGreater(len(errors), 0)
    
    def test_json_schema_validation(self):
        """Test JSON schema validation."""
        config = ProductionConfig()
        
        schema = {
            "type": "object",
            "properties": {
                "optimization": {
                    "type": "object",
                    "properties": {
                        "level": {
                            "type": "string",
                            "enum": ["minimal", "standard", "aggressive", "maximum"]
                        },
                        "max_memory_gb": {
                            "type": "number",
                            "minimum": 0
                        }
                    },
                    "required": ["level", "max_memory_gb"]
                }
            },
            "required": ["optimization"]
        }
        
        config.set_schema(schema)
        
        # Test valid schema
        config.set("optimization.level", "standard")
        config.set("optimization.max_memory_gb", 16.0)
        errors = config.validate_with_schema()
        self.assertEqual(len(errors), 0)
        
        # Test invalid schema
        config.set("optimization.level", "invalid")
        errors = config.validate_with_schema()
        self.assertGreater(len(errors), 0)
    
    def test_config_export(self):
        """Test configuration export functionality."""
        config = ProductionConfig()
        config.set("test.value", "export_test")
        
        # Test JSON export
        json_file = os.path.join(self.temp_dir, "export.json")
        config.export_config(json_file, format="json")
        
        with open(json_file, 'r') as f:
            exported_data = json.load(f)
        
        self.assertEqual(exported_data["test"]["value"], "export_test")
        
        # Test YAML export
        yaml_file = os.path.join(self.temp_dir, "export.yaml")
        config.export_config(yaml_file, format="yaml")
        
        with open(yaml_file, 'r') as f:
            exported_data = yaml.safe_load(f)
        
        self.assertEqual(exported_data["test"]["value"], "export_test")
    
    def test_update_callbacks(self):
        """Test configuration update callbacks."""
        config = ProductionConfig()
        callback_called = []
        
        def test_callback(data):
            callback_called.append(data)
        
        config.add_update_callback(test_callback)
        
        # Trigger update
        config.set("test.value", "callback_test")
        
        # Callback should be called
        self.assertEqual(len(callback_called), 1)
        self.assertIn("test", callback_called[0])
    
    def test_metadata_tracking(self):
        """Test configuration metadata tracking."""
        config = ProductionConfig()
        
        # Test initial metadata
        self.assertIsNotNone(config.config_metadata)
        self.assertEqual(config.config_metadata.source, ConfigSource.DEFAULT)
        self.assertEqual(config.config_metadata.environment, Environment.DEVELOPMENT)
        
        # Test metadata update after change
        config.set("test.value", "metadata_test")
        self.assertEqual(config.config_metadata.source, ConfigSource.API)
    
    def test_checksum_calculation(self):
        """Test configuration checksum calculation."""
        config = ProductionConfig()
        initial_checksum = config.config_metadata.checksum
        
        # Change configuration
        config.set("test.value", "checksum_test")
        new_checksum = config.config_metadata.checksum
        
        # Checksums should be different
        self.assertNotEqual(initial_checksum, new_checksum)
        self.assertIsInstance(new_checksum, str)
        self.assertEqual(len(new_checksum), 32)  # MD5 hash length
    
    def test_thread_safety(self):
        """Test thread safety of configuration operations."""
        import threading
        import time
        
        config = ProductionConfig()
        results = []
        
        def worker(thread_id):
            for i in range(10):
                config.set(f"thread_{thread_id}.value_{i}", f"data_{thread_id}_{i}")
                time.sleep(0.001)  # Small delay to increase chance of race conditions
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All operations should complete without errors
        self.assertEqual(len(config.config_data), 6)  # 5 threads + default sections


class TestFactoryFunctions(unittest.TestCase):
    """Test cases for factory functions."""
    
    def test_create_production_config(self):
        """Test create_production_config factory function."""
        config = create_production_config(environment=Environment.PRODUCTION)
        
        self.assertIsInstance(config, ProductionConfig)
        self.assertEqual(config.environment, Environment.PRODUCTION)
    
    def test_load_config_from_file(self):
        """Test load_config_from_file factory function."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_config = {"test": {"value": "factory_test"}}
            json.dump(test_config, f)
            temp_file = f.name
        
        try:
            config = load_config_from_file(temp_file, Environment.TESTING)
            
            self.assertIsInstance(config, ProductionConfig)
            self.assertEqual(config.environment, Environment.TESTING)
            self.assertEqual(config.get("test.value"), "factory_test")
        finally:
            os.unlink(temp_file)
    
    def test_create_environment_config(self):
        """Test create_environment_config factory function."""
        config = create_environment_config(Environment.STAGING)
        
        self.assertIsInstance(config, ProductionConfig)
        self.assertEqual(config.environment, Environment.STAGING)
    
    def test_production_config_context(self):
        """Test production_config_context context manager."""
        with production_config_context(environment=Environment.PRODUCTION) as config:
            self.assertIsInstance(config, ProductionConfig)
            self.assertEqual(config.environment, Environment.PRODUCTION)
        
        # Config should be cleaned up after context exit
        # (In a real implementation, you might want to verify cleanup)


class TestValidationRules(unittest.TestCase):
    """Test cases for validation rule creation functions."""
    
    def test_create_optimization_validation_rules(self):
        """Test creation of optimization validation rules."""
        rules = create_optimization_validation_rules()
        
        self.assertIsInstance(rules, list)
        self.assertGreater(len(rules), 0)
        
        for rule in rules:
            self.assertIsInstance(rule, ConfigValidationRule)
            self.assertIn("optimization.", rule.field_path)
    
    def test_create_monitoring_validation_rules(self):
        """Test creation of monitoring validation rules."""
        rules = create_monitoring_validation_rules()
        
        self.assertIsInstance(rules, list)
        self.assertGreater(len(rules), 0)
        
        for rule in rules:
            self.assertIsInstance(rule, ConfigValidationRule)
            self.assertIn("monitoring.", rule.field_path)


class TestConfigValidationRule(unittest.TestCase):
    """Test cases for ConfigValidationRule dataclass."""
    
    def test_validation_rule_creation(self):
        """Test ConfigValidationRule creation."""
        rule = ConfigValidationRule(
            field_path="test.field",
            validator=lambda x: x > 0,
            error_message="Value must be positive",
            required=True
        )
        
        self.assertEqual(rule.field_path, "test.field")
        self.assertTrue(rule.required)
        self.assertEqual(rule.error_message, "Value must be positive")
        
        # Test validator function
        self.assertTrue(rule.validator(5))
        self.assertFalse(rule.validator(-1))


class TestConfigMetadata(unittest.TestCase):
    """Test cases for ConfigMetadata dataclass."""
    
    def test_metadata_creation(self):
        """Test ConfigMetadata creation."""
        metadata = ConfigMetadata(
            source=ConfigSource.FILE,
            timestamp=time.time(),
            version="1.0.0",
            environment=Environment.PRODUCTION,
            checksum="test_checksum"
        )
        
        self.assertEqual(metadata.source, ConfigSource.FILE)
        self.assertIsInstance(metadata.timestamp, float)
        self.assertEqual(metadata.version, "1.0.0")
        self.assertEqual(metadata.environment, Environment.PRODUCTION)
        self.assertEqual(metadata.checksum, "test_checksum")


class TestEnvironmentEnum(unittest.TestCase):
    """Test cases for Environment enum."""
    
    def test_environment_values(self):
        """Test Environment enum values."""
        self.assertEqual(Environment.DEVELOPMENT.value, "development")
        self.assertEqual(Environment.STAGING.value, "staging")
        self.assertEqual(Environment.PRODUCTION.value, "production")
        self.assertEqual(Environment.TESTING.value, "testing")


class TestConfigSourceEnum(unittest.TestCase):
    """Test cases for ConfigSource enum."""
    
    def test_config_source_values(self):
        """Test ConfigSource enum values."""
        self.assertEqual(ConfigSource.FILE.value, "file")
        self.assertEqual(ConfigSource.ENVIRONMENT.value, "environment")
        self.assertEqual(ConfigSource.DATABASE.value, "database")
        self.assertEqual(ConfigSource.API.value, "api")
        self.assertEqual(ConfigSource.DEFAULT.value, "default")


if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestProductionConfig,
        TestFactoryFunctions,
        TestValidationRules,
        TestConfigValidationRule,
        TestConfigMetadata,
        TestEnvironmentEnum,
        TestConfigSourceEnum
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")

