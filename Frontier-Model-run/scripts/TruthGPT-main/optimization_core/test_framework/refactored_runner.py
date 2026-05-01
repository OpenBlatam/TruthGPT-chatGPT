"""
Refactored Test Runner
Clean, modular, and maintainable test runner with comprehensive features
"""

import sys
import os
import time
import logging
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add the test framework to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_runner import TestRunner, ExecutionMode
from test_config import TestConfigManager, TestConfig
from test_analytics import TestAnalytics
from test_reporting import TestReporter
from test_metrics import TestMetricsCollector

class RefactoredTestRunner:
    """Refactored test runner with clean architecture."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_manager = TestConfigManager(config_file)
        self.config = self.config_manager.get_config()
        self.logger = self._setup_logger()
        self.analytics = TestAnalytics()
        self.reporter = TestReporter()
        self.metrics_collector = TestMetricsCollector()
        
    def _setup_logger(self):
        """Setup logging for the test runner."""
        logger = logging.getLogger('RefactoredTestRunner')
        logger.setLevel(getattr(logging, self.config.log_level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            if self.config.log_file:
                file_handler = logging.FileHandler(self.config.log_file)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
        
        return logger
    
    def run_tests(self) -> bool:
        """Run tests with the current configuration."""
        self.logger.info("🚀 Starting Refactored Test Runner")
        
        # Validate configuration
        issues = self.config_manager.validate_config()
        if issues:
            self.logger.error(f"Configuration issues: {', '.join(issues)}")
            return False
        
        # Create test runner
        runner = TestRunner(
            verbosity=self.config.verbosity,
            execution_mode=ExecutionMode(self.config.execution_mode.value),
            max_workers=self.config.max_workers,
            output_file=self.config.output_file,
            performance_mode=self.config.performance_mode,
            coverage_mode=self.config.coverage_mode,
            analytics_mode=self.config.analytics_mode,
            intelligent_mode=self.config.intelligent_mode,
            quality_mode=self.config.quality_mode,
            reliability_mode=self.config.reliability_mode,
            optimization_mode=self.config.optimization_mode,
            efficiency_mode=self.config.efficiency_mode,
            scalability_mode=self.config.scalability_mode
        )
        
        # Run tests
        start_time = time.time()
        success = runner.run_tests(
            categories=self.config.categories,
            test_classes=self.config.test_classes,
            priority_filter=self.config.priority_filter,
            tag_filter=self.config.tag_filter,
            optimization_filter=self.config.optimization_filter,
            quality_threshold=self.config.quality_threshold,
            reliability_threshold=self.config.reliability_threshold,
            efficiency_threshold=self.config.efficiency_threshold,
            scalability_threshold=self.config.scalability_threshold
        )
        end_time = time.time()
        
        # Log execution summary
        execution_time = end_time - start_time
        self.logger.info(f"Test execution completed in {execution_time:.2f} seconds")
        self.logger.info(f"Success: {success}")
        
        return success
    
    def generate_report(self, output_file: Optional[str] = None):
        """Generate test report."""
        if not output_file:
            output_file = self.config.output_file
        
        # Generate report based on format
        if self.config.output_format == 'html':
            self.reporter.generate_html_report({}, output_file)
        elif self.config.output_format == 'csv':
            self.reporter.generate_csv_report({}, output_file)
        elif self.config.output_format == 'markdown':
            self.reporter.generate_markdown_report({}, output_file)
        else:
            self.reporter.save_comprehensive_report({}, output_file)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return self.config_manager.get_config_summary()
    
    def update_config(self, **kwargs):
        """Update configuration."""
        self.config_manager.update_config(**kwargs)
        self.config = self.config_manager.get_config()
    
    def save_config(self, config_file: str):
        """Save current configuration."""
        self.config_manager.save_config(config_file)
    
    def create_sample_config(self, output_file: str):
        """Create sample configuration file."""
        self.config_manager.create_sample_config(output_file)

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Refactored Test Runner for Optimization Core')
    
    # Configuration options
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--execution-mode', choices=['sequential', 'parallel', 'distributed', 'adaptive', 'intelligent', 'ultra_intelligent'], 
                       help='Test execution mode')
    parser.add_argument('--workers', type=int, help='Number of parallel workers')
    parser.add_argument('--verbosity', type=int, choices=[0, 1, 2, 3], help='Test verbosity level')
    parser.add_argument('--timeout', type=int, help='Test timeout in seconds')
    
    # Output options
    parser.add_argument('--output', type=str, help='Output file for test report')
    parser.add_argument('--format', choices=['json', 'html', 'csv', 'markdown'], help='Output format')
    
    # Filtering options
    parser.add_argument('--categories', nargs='+', help='Test categories to run')
    parser.add_argument('--test-classes', nargs='+', help='Specific test classes to run')
    parser.add_argument('--priority', choices=['critical', 'high', 'medium', 'low', 'optional', 'experimental'], 
                       help='Filter by priority level')
    parser.add_argument('--tags', nargs='+', help='Filter by tags')
    parser.add_argument('--optimization', choices=['quantum', 'evolutionary', 'meta_learning', 'hyperparameter', 'neural_architecture', 'ultra_advanced', 'ultimate', 'bulk'], 
                       help='Filter by optimization type')
    
    # Threshold options
    parser.add_argument('--quality-threshold', type=float, help='Quality threshold for filtering')
    parser.add_argument('--reliability-threshold', type=float, help='Reliability threshold for filtering')
    parser.add_argument('--performance-threshold', type=float, help='Performance threshold for filtering')
    parser.add_argument('--optimization-threshold', type=float, help='Optimization threshold for filtering')
    parser.add_argument('--efficiency-threshold', type=float, help='Efficiency threshold for filtering')
    parser.add_argument('--scalability-threshold', type=float, help='Scalability threshold for filtering')
    
    # Feature flags
    parser.add_argument('--performance', action='store_true', help='Enable performance mode')
    parser.add_argument('--coverage', action='store_true', help='Enable coverage mode')
    parser.add_argument('--analytics', action='store_true', help='Enable analytics mode')
    parser.add_argument('--intelligent', action='store_true', help='Enable intelligent mode')
    parser.add_argument('--quality', action='store_true', help='Enable quality mode')
    parser.add_argument('--reliability', action='store_true', help='Enable reliability mode')
    parser.add_argument('--optimization', action='store_true', help='Enable optimization mode')
    parser.add_argument('--efficiency', action='store_true', help='Enable efficiency mode')
    parser.add_argument('--scalability', action='store_true', help='Enable scalability mode')
    
    # Utility options
    parser.add_argument('--create-config', type=str, help='Create sample configuration file')
    parser.add_argument('--config-summary', action='store_true', help='Show configuration summary')
    parser.add_argument('--validate-config', action='store_true', help='Validate configuration')
    
    args = parser.parse_args()
    
    # Create test runner
    runner = RefactoredTestRunner(args.config)
    
    # Handle utility options
    if args.create_config:
        runner.create_sample_config(args.create_config)
        print(f"Sample configuration created: {args.create_config}")
        return
    
    if args.config_summary:
        summary = runner.get_config_summary()
        print("Configuration Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        return
    
    if args.validate_config:
        issues = runner.config_manager.validate_config()
        if issues:
            print("Configuration issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("Configuration is valid")
        return
    
    # Update configuration from command line arguments
    config_updates = {}
    if args.execution_mode:
        config_updates['execution_mode'] = args.execution_mode
    if args.workers:
        config_updates['max_workers'] = args.workers
    if args.verbosity is not None:
        config_updates['verbosity'] = args.verbosity
    if args.timeout:
        config_updates['timeout'] = args.timeout
    if args.output:
        config_updates['output_file'] = args.output
    if args.format:
        config_updates['output_format'] = args.format
    if args.categories:
        config_updates['categories'] = args.categories
    if args.test_classes:
        config_updates['test_classes'] = args.test_classes
    if args.priority:
        config_updates['priority_filter'] = args.priority
    if args.tags:
        config_updates['tag_filter'] = args.tags
    if args.optimization:
        config_updates['optimization_filter'] = args.optimization
    if args.quality_threshold:
        config_updates['quality_threshold'] = args.quality_threshold
    if args.reliability_threshold:
        config_updates['reliability_threshold'] = args.reliability_threshold
    if args.performance_threshold:
        config_updates['performance_threshold'] = args.performance_threshold
    if args.optimization_threshold:
        config_updates['optimization_threshold'] = args.optimization_threshold
    if args.efficiency_threshold:
        config_updates['efficiency_threshold'] = args.efficiency_threshold
    if args.scalability_threshold:
        config_updates['scalability_threshold'] = args.scalability_threshold
    if args.performance:
        config_updates['performance_mode'] = True
    if args.coverage:
        config_updates['coverage_mode'] = True
    if args.analytics:
        config_updates['analytics_mode'] = True
    if args.intelligent:
        config_updates['intelligent_mode'] = True
    if args.quality:
        config_updates['quality_mode'] = True
    if args.reliability:
        config_updates['reliability_mode'] = True
    if args.optimization:
        config_updates['optimization_mode'] = True
    if args.efficiency:
        config_updates['efficiency_mode'] = True
    if args.scalability:
        config_updates['scalability_mode'] = True
    
    if config_updates:
        runner.update_config(**config_updates)
    
    # Run tests
    success = runner.run_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()











