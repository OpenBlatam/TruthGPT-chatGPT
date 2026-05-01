from typing import Any, Callable, Dict


_DATASET_BUILDERS: Dict[str, Callable[[Dict[str, Any]], Any]] = {}


def register_dataset(name: str):
    def _wrap(fn: Callable[[Dict[str, Any]], Any]):
        _DATASET_BUILDERS[name] = fn
        return fn

    return _wrap


def build_dataset(name: str, cfg: Dict[str, Any]):
    if name not in _DATASET_BUILDERS:
        raise KeyError(f"Dataset '{name}' is not registered. Available: {list(_DATASET_BUILDERS.keys())}")
    return _DATASET_BUILDERS[name](cfg)






