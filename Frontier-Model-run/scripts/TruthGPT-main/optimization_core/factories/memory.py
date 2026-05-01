from optimization_core.factories.registry import Registry
from optimization_core.modules.memory.advanced_memory_manager import create_advanced_memory_manager, create_memory_config

MEMORY_MANAGERS = Registry()


@MEMORY_MANAGERS.register("adaptive")
def build_adaptive(**kwargs):
    cfg = create_memory_config(**kwargs)
    return create_advanced_memory_manager(cfg)


@MEMORY_MANAGERS.register("static")
def build_static(**kwargs):
    cfg = create_memory_config(use_xformers=False, prefer_bf16=False, **kwargs)
    return create_advanced_memory_manager(cfg)






