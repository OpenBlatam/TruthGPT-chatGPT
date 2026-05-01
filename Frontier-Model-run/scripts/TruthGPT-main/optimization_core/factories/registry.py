from typing import Callable, Dict, Type, Any


class Registry:
    def __init__(self) -> None:
        self._items: Dict[str, Any] = {}

    def register(self, name: str) -> Callable[[Any], Any]:
        def deco(obj: Any) -> Any:
            self._items[name] = obj
            return obj
        return deco

    def get(self, name: str) -> Any:
        if name not in self._items:
            raise KeyError(f"Registry item '{name}' not found")
        return self._items[name]

    def build(self, name: str, *args, **kwargs) -> Any:
        cls_or_fn = self.get(name)
        return cls_or_fn(*args, **kwargs)






