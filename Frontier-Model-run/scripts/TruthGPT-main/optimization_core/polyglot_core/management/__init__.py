"""
Management modules for polyglot_core.

Configuration, migration, version management, plugins, CLI, and documentation.
"""

from ..config import (
    PolyglotConfig,
    ConfigManager,
    Environment,
    get_config_manager,
    get_config,
    load_config,
    save_config,
)

from ..migration import (
    Migration,
    MigrationManager,
    get_migration_manager,
    register_migration,
)

from ..version import (
    Version,
    get_version,
    get_version_info,
    check_compatibility,
)

from ..plugins import (
    Plugin,
    PluginManager,
    get_plugin_manager,
    register_plugin,
    get_plugin,
)

from ..cli import (
    create_cli_parser,
    main_cli,
)

from ..docs import (
    DocumentationGenerator,
    get_documentation_generator,
    generate_docs,
)

__all__ = [
    # Config
    "PolyglotConfig",
    "ConfigManager",
    "Environment",
    "get_config_manager",
    "get_config",
    "load_config",
    "save_config",
    # Migration
    "Migration",
    "MigrationManager",
    "get_migration_manager",
    "register_migration",
    # Version
    "Version",
    "get_version",
    "get_version_info",
    "check_compatibility",
    # Plugins
    "Plugin",
    "PluginManager",
    "get_plugin_manager",
    "register_plugin",
    "get_plugin",
    # CLI
    "create_cli_parser",
    "main_cli",
    # Documentation
    "DocumentationGenerator",
    "get_documentation_generator",
    "generate_docs",
]













