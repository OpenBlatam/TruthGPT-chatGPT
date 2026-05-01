# Core Directory Organization - Refactoring Summary

## Overview

This document describes the organization of the `core/` directory to improve code discoverability and maintainability by creating logical subdirectories with unified interfaces.

## Completed Refactorings

### 1. ✅ Created Systems Module

**Location:** `core/systems/__init__.py`

**Purpose:** Organize core infrastructure systems

**Exports:**
- **Dynamic Factory**: `DynamicFactory`, `factory`, `register_component`, `create_factory`
- **Event System**: `EventEmitter`, `EventType`, `Event`, `get_event_emitter`, `emit_event`, `on_event`
- **Service Registry**: `ServiceRegistry`, `ServiceContainer`, `register_service`, `get_service`
- **Plugin System**: `Plugin`, `PluginManager`, `get_plugin_manager`
- **Module Loader**: `ModuleLoader`, `get_module_loader`, `lazy_load`

**Discovery Functions:**
- `list_available_systems()` - List all available core systems
- `get_system_info(system_name)` - Get system information

### 2. ✅ Updated Main Core Module

**Location:** `core/__init__.py`

**Changes:**
- Added lazy import system for all submodules
- Maintained backward compatibility with direct imports
- Added `list_available_core_modules()` function
- All submodules accessible via `core.systems`, `core.optimizers`, etc.

**Submodules Available:**
- `systems` - Infrastructure systems
- `optimizers` - Core optimizers
- `services` - Service implementations
- `adapters` - Adapter components
- `framework` - Framework components
- `validation` - Validation components
- `composition` - Composition and workflow builders
- `runtime` - Runtime components

## Usage Examples

### Accessing Systems Components

```python
# New organized way (recommended)
from optimization_core.core.systems import (
    DynamicFactory,
    EventEmitter,
    ServiceRegistry,
    PluginManager,
    ModuleLoader,
)

# Or via main core module
from optimization_core.core import systems
factory = systems.DynamicFactory()

# Discovery
from optimization_core.core.systems import list_available_systems
available = list_available_systems()
```

### Backward Compatible Access

```python
# These still work (backward compatible)
from optimization_core.core import (
    DynamicFactory,
    EventEmitter,
    ServiceRegistry,
    PluginManager,
    ModuleLoader,
    ServiceContainer,
    register_service,
    get_service,
    EventType,
    Event,
    get_event_emitter,
    emit_event,
    on_event,
    Plugin,
    get_plugin_manager,
    factory,
    register_component,
    create_factory,
    get_module_loader,
    lazy_load,
)
```

### Accessing Other Core Submodules

```python
# Access optimizers
from optimization_core.core import optimizers
optimizer = optimizers.create_core_optimizer("extreme", config)

# Access services
from optimization_core.core import services
service = services.BaseService()

# Access validation
from optimization_core.core import validation
validator = validation.Validator()

# Access composition
from optimization_core.core import composition
assembler = composition.ComponentAssembler()
```

## Backward Compatibility

**100% Backward Compatible**

All existing imports continue to work:

```python
# These still work:
from optimization_core.core import DynamicFactory
from optimization_core.core import EventEmitter
from optimization_core.core import ServiceRegistry
from optimization_core.core import PluginManager
from optimization_core.core import ModuleLoader
from optimization_core.core.service_registry import ServiceRegistry
from optimization_core.core.event_system import EventEmitter
```

## Benefits

1. **Better Organization**: Infrastructure systems grouped logically
2. **Improved Discoverability**: Easy to find specific systems
3. **Unified Interfaces**: Consistent API across all core modules
4. **Lazy Loading**: Fast startup with lazy imports for submodules
5. **Discovery Functions**: Programmatic access to available systems
6. **Backward Compatibility**: All existing code continues to work
7. **Maintainability**: Clear structure for future additions

## Statistics

- **New Subdirectories**: 1 (systems)
- **New `__init__.py` Files**: 1
- **Discovery Functions**: 2
- **Backward Compatibility**: 100%
- **Linter Errors**: 0

## System Components

### Dynamic Factory
- **DynamicFactory**: Dynamic factory with automatic registration
- **factory**: Decorator for factory registration
- **register_component**: Register a component
- **create_factory**: Create a factory instance

### Event System
- **EventEmitter**: Event emitter for decoupled communication
- **EventType**: Standard event types enum
- **Event**: Event data structure
- **get_event_emitter**: Get global event emitter
- **emit_event**: Emit an event
- **on_event**: Register event handler

### Service Registry
- **ServiceRegistry**: Central registry for services
- **ServiceContainer**: Service container with dependency injection
- **register_service**: Register a service
- **get_service**: Get a registered service

### Plugin System
- **Plugin**: Base class for plugins
- **PluginManager**: Plugin manager for dynamic loading
- **get_plugin_manager**: Get global plugin manager

### Module Loader
- **ModuleLoader**: Lazy module loader
- **get_module_loader**: Get global module loader
- **lazy_load**: Lazy load a module

## Future Enhancements (Optional)

1. ⏳ Consider physically moving files to subdirectories (currently using lazy imports)
2. ⏳ Add more examples to documentation
3. ⏳ Create factory functions for each system category
4. ⏳ Add type hints to all discovery functions
5. ⏳ Create unified configuration system for core components

---

**Date**: 2024  
**Version**: 4.5.0 (Core Organization Refactoring)  
**Status**: ✅ Complete

**This refactoring organizes the core directory into logical submodules while maintaining 100% backward compatibility!**

