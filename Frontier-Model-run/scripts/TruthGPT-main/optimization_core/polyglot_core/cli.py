"""
CLI utilities for polyglot_core.

Provides command-line interface for common operations.
"""

import argparse
import sys
from typing import Optional
from pathlib import Path


def create_cli_parser() -> argparse.ArgumentParser:
    """
    Create CLI argument parser.
    
    Returns:
        ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Polyglot Core CLI - High-performance multi-language backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check backends
  polyglot check-backends
  
  # Run benchmarks
  polyglot benchmark --quick
  
  # Health check
  polyglot health
  
  # Generate report
  polyglot benchmark --output results.json --report report.html
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Check backends command
    check_parser = subparsers.add_parser('check-backends', help='Check backend availability')
    check_parser.add_argument('--detailed', action='store_true', help='Show detailed information')
    check_parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
    benchmark_parser.add_argument('--quick', action='store_true', help='Run quick benchmarks')
    benchmark_parser.add_argument('--output', type=str, help='Output file for results (JSON)')
    benchmark_parser.add_argument('--report', type=str, help='Generate HTML report')
    
    # Health check command
    health_parser = subparsers.add_parser('health', help='Check system health')
    health_parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_parser.add_argument('--show', action='store_true', help='Show current configuration')
    config_parser.add_argument('--load', type=str, help='Load configuration from file')
    config_parser.add_argument('--save', type=str, help='Save configuration to file')
    config_parser.add_argument('--env', type=str, choices=['dev', 'prod', 'staging'], help='Set environment')
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show version information')
    
    return parser


def main_cli(args: Optional[list] = None):
    """
    Main CLI entry point.
    
    Args:
        args: Command line arguments (default: sys.argv[1:])
    """
    parser = create_cli_parser()
    parsed_args = parser.parse_args(args)
    
    if not parsed_args.command:
        parser.print_help()
        return
    
    # Add parent to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    try:
        if parsed_args.command == 'check-backends':
            from .scripts.check_backends import main
            sys.argv = ['check_backends'] + (['--detailed'] if parsed_args.detailed else [])
            if parsed_args.json:
                sys.argv.append('--json')
            main()
        
        elif parsed_args.command == 'benchmark':
            from .scripts.run_benchmarks import main
            sys.argv = ['run_benchmarks'] + (['--quick'] if parsed_args.quick else [])
            if parsed_args.output:
                sys.argv.extend(['--output', parsed_args.output])
            if parsed_args.report:
                sys.argv.extend(['--report', parsed_args.report])
            main()
        
        elif parsed_args.command == 'health':
            from .scripts.health_check import main
            sys.argv = ['health_check'] + (['--json'] if parsed_args.json else [])
            main()
        
        elif parsed_args.command == 'config':
            from .config import get_config_manager, PolyglotConfig, Environment
            
            manager = get_config_manager()
            
            if parsed_args.show:
                config = manager.get_config()
                if config:
                    print(config.to_yaml())
                else:
                    print("No configuration loaded")
            
            elif parsed_args.load:
                config = manager.load_config(parsed_args.load)
                print(f"Configuration loaded from {parsed_args.load}")
            
            elif parsed_args.save:
                config = manager.get_config() or PolyglotConfig.default()
                manager.save_config(config, parsed_args.save)
                print(f"Configuration saved to {parsed_args.save}")
            
            elif parsed_args.env:
                env_map = {
                    'dev': Environment.DEVELOPMENT,
                    'prod': Environment.PRODUCTION,
                    'staging': Environment.STAGING
                }
                config = manager.get_config() or PolyglotConfig.default()
                config.environment = env_map[parsed_args.env]
                manager.save_config(config)
                print(f"Environment set to {parsed_args.env}")
            
            else:
                config_parser.print_help()
        
        elif parsed_args.command == 'version':
            from . import __version__
            print(f"Polyglot Core version {__version__}")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main_cli()













