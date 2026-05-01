"""
Test Configuration Framework
Centralized configuration management for test execution
"""

import os
import json
import yaml
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

class ExecutionMode(Enum):
    """Test execution modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DISTRIBUTED = "distributed"
    ADAPTIVE = "adaptive"
    INTELLIGENT = "intelligent"
    ULTRA_INTELLIGENT = "ultra_intelligent"

class TestPriority(Enum):
    """Test priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    OPTIONAL = "optional"
    EXPERIMENTAL = "experimental"

class TestCategory(Enum):
    """Test categories."""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    COMPATIBILITY = "compatibility"
    ADVANCED = "advanced"
    QUANTUM = "quantum"
    EVOLUTIONARY = "evolutionary"
    META_LEARNING = "meta_learning"
    HYPERPARAMETER = "hyperparameter"
    NEURAL_ARCHITECTURE = "neural_architecture"
    ULTRA_ADVANCED = "ultra_advanced"
    ULTIMATE = "ultimate"
    BULK = "bulk"
    LIBRARY = "library"

@dataclass
class TestConfig:
    """Test configuration settings."""
    # Execution settings
    execution_mode: ExecutionMode = ExecutionMode.ULTRA_INTELLIGENT
    max_workers: int = None
    verbosity: int = 2
    timeout: int = 300
    
    # Output settings
    output_file: Optional[str] = None
    output_format: str = "json"  # json, html, csv, markdown
    
    # Filtering settings
    categories: List[str] = field(default_factory=list)
    test_classes: List[str] = field(default_factory=list)
    priority_filter: Optional[str] = None
    tag_filter: List[str] = field(default_factory=list)
    optimization_filter: Optional[str] = None
    
    # Threshold settings
    quality_threshold: float = 0.8
    reliability_threshold: float = 0.8
    performance_threshold: float = 0.8
    optimization_threshold: float = 0.8
    efficiency_threshold: float = 0.8
    scalability_threshold: float = 0.8
    
    # Feature flags
    performance_mode: bool = False
    coverage_mode: bool = False
    analytics_mode: bool = False
    intelligent_mode: bool = False
    quality_mode: bool = False
    reliability_mode: bool = False
    optimization_mode: bool = False
    efficiency_mode: bool = False
    scalability_mode: bool = False
    
    # System settings
    log_level: str = "INFO"
    log_file: Optional[str] = None
    debug_mode: bool = False
    
    # Advanced settings
    retry_count: int = 3
    parallel: bool = True
    resource_monitoring: bool = True
    trend_analysis: bool = True
    recommendation_engine: bool = True
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.max_workers is None:
            self.max_workers = os.cpu_count()
        
        if self.output_file is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file = f"test_report_{timestamp}.json"

class TestConfigManager:
    """Test configuration manager."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config_file = config_file
        self.config = TestConfig()
        
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def load_config(self, config_file: str):
        """Load configuration from file."""
        try:
            with open(config_file, 'r') as f:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            self._apply_config_data(config_data)
            self.logger.info(f"Configuration loaded from {config_file}")
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
    
    def save_config(self, config_file: str):
        """Save configuration to file."""
        try:
            config_data = self._get_config_data()
            
            with open(config_file, 'w') as f:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    yaml.dump(config_data, f, default_flow_style=False)
                else:
                    json.dump(config_data, f, indent=2)
            
            self.logger.info(f"Configuration saved to {config_file}")
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
    
    def _apply_config_data(self, config_data: Dict[str, Any]):
        """Apply configuration data to config object."""
        for key, value in config_data.items():
            if hasattr(self.config, key):
                if key == 'execution_mode':
                    self.config.execution_mode = ExecutionMode(value)
                elif key == 'priority_filter':
                    self.config.priority_filter = value
                elif key == 'optimization_filter':
                    self.config.optimization_filter = value
                else:
                    setattr(self.config, key, value)
    
    def _get_config_data(self) -> Dict[str, Any]:
        """Get configuration data as dictionary."""
        config_data = {}
        for field_name, field_value in self.config.__dict__.items():
            if isinstance(field_value, Enum):
                config_data[field_name] = field_value.value
            else:
                config_data[field_name] = field_value
        return config_data
    
    def get_config(self) -> TestConfig:
        """Get current configuration."""
        return self.config
    
    def update_config(self, **kwargs):
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                if key == 'execution_mode':
                    self.config.execution_mode = ExecutionMode(value)
                else:
                    setattr(self.config, key, value)
            else:
                self.logger.warning(f"Unknown configuration key: {key}")
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate execution mode
        if not isinstance(self.config.execution_mode, ExecutionMode):
            issues.append("Invalid execution mode")
        
        # Validate max workers
        if self.config.max_workers <= 0:
            issues.append("Max workers must be positive")
        
        # Validate verbosity
        if self.config.verbosity < 0 or self.config.verbosity > 3:
            issues.append("Verbosity must be between 0 and 3")
        
        # Validate timeout
        if self.config.timeout <= 0:
            issues.append("Timeout must be positive")
        
        # Validate thresholds
        thresholds = [
            ('quality_threshold', self.config.quality_threshold),
            ('reliability_threshold', self.config.reliability_threshold),
            ('performance_threshold', self.config.performance_threshold),
            ('optimization_threshold', self.config.optimization_threshold),
            ('efficiency_threshold', self.config.efficiency_threshold),
            ('scalability_threshold', self.config.scalability_threshold)
        ]
        
        for name, value in thresholds:
            if not 0.0 <= value <= 1.0:
                issues.append(f"{name} must be between 0.0 and 1.0")
        
        # Validate output format
        valid_formats = ['json', 'html', 'csv', 'markdown']
        if self.config.output_format not in valid_formats:
            issues.append(f"Output format must be one of: {', '.join(valid_formats)}")
        
        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.config.log_level not in valid_log_levels:
            issues.append(f"Log level must be one of: {', '.join(valid_log_levels)}")
        
        return issues
    
    def get_default_config(self) -> TestConfig:
        """Get default configuration."""
        return TestConfig()
    
    def create_sample_config(self, output_file: str):
        """Create sample configuration file."""
        sample_config = {
            "execution_mode": "ultra_intelligent",
            "max_workers": 32,
            "verbosity": 2,
            "timeout": 300,
            "output_file": "test_report.json",
            "output_format": "json",
            "categories": [],
            "test_classes": [],
            "priority_filter": None,
            "tag_filter": [],
            "optimization_filter": None,
            "quality_threshold": 0.8,
            "reliability_threshold": 0.8,
            "performance_threshold": 0.8,
            "optimization_threshold": 0.8,
            "efficiency_threshold": 0.8,
            "scalability_threshold": 0.8,
            "performance_mode": False,
            "coverage_mode": False,
            "analytics_mode": False,
            "intelligent_mode": False,
            "quality_mode": False,
            "reliability_mode": False,
            "optimization_mode": False,
            "efficiency_mode": False,
            "scalability_mode": False,
            "log_level": "INFO",
            "log_file": None,
            "debug_mode": False,
            "retry_count": 3,
            "parallel": True,
            "resource_monitoring": True,
            "trend_analysis": True,
            "recommendation_engine": True
        }
        
        try:
            with open(output_file, 'w') as f:
                if output_file.endswith('.yaml') or output_file.endswith('.yml'):
                    yaml.dump(sample_config, f, default_flow_style=False)
                else:
                    json.dump(sample_config, f, indent=2)
            
            self.logger.info(f"Sample configuration created: {output_file}")
        except Exception as e:
            self.logger.error(f"Error creating sample configuration: {e}")
    
    def merge_configs(self, *configs: TestConfig) -> TestConfig:
        """Merge multiple configurations."""
        merged_config = TestConfig()
        
        for config in configs:
            for field_name, field_value in config.__dict__.items():
                if field_value is not None:
                    setattr(merged_config, field_name, field_value)
        
        return merged_config
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            'execution_mode': self.config.execution_mode.value,
            'max_workers': self.config.max_workers,
            'verbosity': self.config.verbosity,
            'timeout': self.config.timeout,
            'output_file': self.config.output_file,
            'output_format': self.config.output_format,
            'categories': self.config.categories,
            'test_classes': self.config.test_classes,
            'priority_filter': self.config.priority_filter,
            'tag_filter': self.config.tag_filter,
            'optimization_filter': self.config.optimization_filter,
            'quality_threshold': self.config.quality_threshold,
            'reliability_threshold': self.config.reliability_threshold,
            'performance_threshold': self.config.performance_threshold,
            'optimization_threshold': self.config.optimization_threshold,
            'efficiency_threshold': self.config.efficiency_threshold,
            'scalability_threshold': self.config.scalability_threshold,
            'feature_flags': {
                'performance_mode': self.config.performance_mode,
                'coverage_mode': self.config.coverage_mode,
                'analytics_mode': self.config.analytics_mode,
                'intelligent_mode': self.config.intelligent_mode,
                'quality_mode': self.config.quality_mode,
                'reliability_mode': self.config.reliability_mode,
                'optimization_mode': self.config.optimization_mode,
                'efficiency_mode': self.config.efficiency_mode,
                'scalability_mode': self.config.scalability_mode
            },
            'system_settings': {
                'log_level': self.config.log_level,
                'log_file': self.config.log_file,
                'debug_mode': self.config.debug_mode
            },
            'advanced_settings': {
                'retry_count': self.config.retry_count,
                'parallel': self.config.parallel,
                'resource_monitoring': self.config.resource_monitoring,
                'trend_analysis': self.config.trend_analysis,
                'recommendation_engine': self.config.recommendation_engine
            }
        }











