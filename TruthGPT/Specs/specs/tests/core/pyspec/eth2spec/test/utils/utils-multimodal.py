from typing import Dict, Any, Callable, Generator, Tuple
import base64

def multimodal_vector_test(description: str = None):
    """
    Like Ethereum's vector_test, this handles generator vs. test mode for multimodal AI tests.
    """
    def wrapper(fn: Callable[..., Generator]) -> Callable[..., Any]:
        def entry(*args, **kwargs):
            def generator_mode():
                if description:
                    yield "description", "meta", description

                for data in fn(*args, **kwargs):
                    if len(data) != 2:
                        yield data  # Already well-formed
                        continue
                    key, value = data
                    if value is None:
                        continue
                    if isinstance(value, bytes):
                        # Raw binary: encode as base64
                        yield key, "binary", base64.b64encode(value).decode()
                    elif isinstance(value, str):
                        yield key, "text", value
                    elif isinstance(value, dict):
                        yield key, "data", value
                    elif isinstance(value, list):
                        yield key, "list", value
                        yield f"{key}_count", "meta", len(value)
                    else:
                        yield key, "data", value

            if kwargs.pop("generator_mode", False):
                return generator_mode()
            else:
                for _ in fn(*args, **kwargs):
                    pass
                return None
        return entry
    return wrapper


def with_tags(tags: Dict[str, Any]):
    """
    Adds metadata tags to the vector generation output.
    """
    def wrapper(fn: Callable[..., Generator]) -> Callable[..., Generator]:
        def entry(*args, **kwargs):
            yielded_any = False
            for item in fn(*args, **kwargs):
                yield item
                yielded_any = True
            if yielded_any:
                for k, v in tags.items():
                    yield k, "meta", v
        return entry
    return wrapper
