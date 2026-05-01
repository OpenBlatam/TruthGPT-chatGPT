"""
Plugin System for TruthGPT Compiler
Extensible plugin architecture for compiler components
"""

from .plugin_system import (
    CompilerPlugin, PluginManager, PluginRegistry, PluginInterface,
    PluginConfig, PluginResult, PluginStatus,
    create_plugin_manager, plugin_compilation_context
)

__all__ = [
    'CompilerPlugin',
    'PluginManager',
    'PluginRegistry',
    'PluginInterface',
    'PluginConfig',
    'PluginResult',
    'PluginStatus',
    'create_plugin_manager',
    'plugin_compilation_context',
]






