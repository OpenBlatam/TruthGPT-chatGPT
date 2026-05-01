"""
Common collection utilities for optimization_core.

Provides reusable functions for working with lists, dicts, and collections.
"""

from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
from collections import defaultdict

from .types import DictStrAny, T, K, V


# ════════════════════════════════════════════════════════════════════════════════
# CHUNKING AND BATCHING
# ════════════════════════════════════════════════════════════════════════════════

def chunk_list(items: List[T], chunk_size: int) -> List[List[T]]:
    """
    Split list into chunks of specified size.
    
    Args:
        items: List to chunk (must be a list)
        chunk_size: Size of each chunk (must be positive)
    
    Returns:
        List of chunks (empty list if items is empty)
    
    Raises:
        TypeError: If items is not a list
        ValueError: If chunk_size <= 0
    
    Examples:
        >>> chunk_list([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
        >>> chunk_list([], 5)
        []
    """
    # Validate inputs
    if not isinstance(items, list):
        raise TypeError(f"items must be a list, got {type(items).__name__}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    
    # Handle empty list efficiently
    if not items:
        return []
    
    # Create chunks using list comprehension (efficient slicing)
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def chunk_list_iter(items: List[T], chunk_size: int) -> Iterator[List[T]]:
    """
    Split list into chunks (iterator version for memory efficiency).
    
    Args:
        items: List to chunk (must be a list)
        chunk_size: Size of each chunk (must be positive)
    
    Yields:
        Chunks (one at a time to save memory)
    
    Raises:
        TypeError: If items is not a list
        ValueError: If chunk_size <= 0
    
    Example:
        >>> for chunk in chunk_list_iter([1, 2, 3, 4, 5], 2):
        ...     print(chunk)
        [1, 2]
        [3, 4]
        [5]
    """
    # Validate inputs
    if not isinstance(items, list):
        raise TypeError(f"items must be a list, got {type(items).__name__}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    
    # Yield chunks one at a time (memory efficient for large lists)
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


# ════════════════════════════════════════════════════════════════════════════════
# FLATTENING
# ════════════════════════════════════════════════════════════════════════════════

def flatten_list(nested_list: List[List[T]]) -> List[T]:
    """
    Flatten nested list (one level deep).
    
    Args:
        nested_list: List of lists (must be a list)
    
    Returns:
        Flattened list (empty list if nested_list is empty)
    
    Raises:
        TypeError: If nested_list is not a list
    
    Examples:
        >>> flatten_list([[1, 2], [3, 4], [5]])
        [1, 2, 3, 4, 5]
        >>> flatten_list([])
        []
    """
    # Validate input
    if not isinstance(nested_list, list):
        raise TypeError(f"nested_list must be a list, got {type(nested_list).__name__}")
    
    # Handle empty list efficiently
    if not nested_list:
        return []
    
    # Flatten using list comprehension (efficient for one-level nesting)
    return [item for sublist in nested_list for item in sublist]


def flatten_list_deep(nested_list: List[Any]) -> List[Any]:
    """
    Flatten deeply nested list recursively (handles arbitrary nesting depth).
    
    Args:
        nested_list: Deeply nested list (must be a list)
    
    Returns:
        Flattened list (empty list if nested_list is empty)
    
    Raises:
        TypeError: If nested_list is not a list
    
    Examples:
        >>> flatten_list_deep([[1, [2, 3]], [4, [5]]])
        [1, 2, 3, 4, 5]
        >>> flatten_list_deep([1, 2, 3])
        [1, 2, 3]
        >>> flatten_list_deep([])
        []
    """
    # Validate input
    if not isinstance(nested_list, list):
        raise TypeError(f"nested_list must be a list, got {type(nested_list).__name__}")
    
    # Handle empty list efficiently
    if not nested_list:
        return []
    
    # Recursively flatten nested structures
    result = []
    for item in nested_list:
        if isinstance(item, list):
            # Recursively flatten nested lists
            result.extend(flatten_list_deep(item))
        else:
            # Add non-list items directly
            result.append(item)
    return result


# ════════════════════════════════════════════════════════════════════════════════
# GROUPING
# ════════════════════════════════════════════════════════════════════════════════

def group_by(
    items: List[T],
    key_func: Callable[[T], K]
) -> Dict[K, List[T]]:
    """
    Group items by key function.
    
    Args:
        items: List of items (must be a list)
        key_func: Function to extract key from item (must be callable)
    
    Returns:
        Dictionary mapping keys to lists of items (empty dict if items is empty)
    
    Raises:
        TypeError: If items is not a list or key_func is not callable
    
    Examples:
        >>> group_by([1, 2, 3, 4, 5], lambda x: x % 2)
        {0: [2, 4], 1: [1, 3, 5]}
        >>> group_by([], lambda x: x)
        {}
    """
    # Validate inputs
    if not isinstance(items, list):
        raise TypeError(f"items must be a list, got {type(items).__name__}")
    if not callable(key_func):
        raise TypeError(f"key_func must be callable, got {type(key_func).__name__}")
    
    # Handle empty list efficiently
    if not items:
        return {}
    
    # Group items using defaultdict for efficiency
    grouped = defaultdict(list)
    for item in items:
        key = key_func(item)
        grouped[key].append(item)
    return dict(grouped)


def group_by_key(
    items: List[DictStrAny],
    key: str
) -> Dict[Any, List[DictStrAny]]:
    """
    Group dictionary items by key.
    
    Args:
        items: List of dictionaries (must be a list)
        key: Key to group by (must be a non-empty string)
    
    Returns:
        Dictionary mapping key values to lists of items (empty dict if items is empty)
    
    Raises:
        TypeError: If items is not a list or key is not a string
        ValueError: If key is empty
    
    Examples:
        >>> group_by_key([{"type": "A", "val": 1}, {"type": "B", "val": 2}], "type")
        {'A': [{'type': 'A', 'val': 1}], 'B': [{'type': 'B', 'val': 2}]}
        >>> group_by_key([], "type")
        {}
    """
    # Validate inputs
    if not isinstance(items, list):
        raise TypeError(f"items must be a list, got {type(items).__name__}")
    if not isinstance(key, str):
        raise TypeError(f"key must be a string, got {type(key).__name__}")
    if not key:
        raise ValueError("key cannot be empty")
    
    # Handle empty list efficiently
    if not items:
        return {}
    
    # Use group_by with a lambda that extracts the key value
    return group_by(items, lambda x: x.get(key))


# ════════════════════════════════════════════════════════════════════════════════
# PARTITIONING
# ════════════════════════════════════════════════════════════════════════════════

def partition(
    items: List[T],
    predicate: Callable[[T], bool]
) -> Tuple[List[T], List[T]]:
    """
    Partition items into two lists based on predicate.
    
    Args:
        items: List of items (must be a list)
        predicate: Function that returns True/False (must be callable)
    
    Returns:
        Tuple of (items where predicate is True, items where predicate is False)
        Both lists are empty if items is empty
    
    Raises:
        TypeError: If items is not a list or predicate is not callable
    
    Examples:
        >>> partition([1, 2, 3, 4, 5], lambda x: x % 2 == 0)
        ([2, 4], [1, 3, 5])
        >>> partition([], lambda x: x > 0)
        ([], [])
    """
    # Validate inputs
    if not isinstance(items, list):
        raise TypeError(f"items must be a list, got {type(items).__name__}")
    if not callable(predicate):
        raise TypeError(f"predicate must be callable, got {type(predicate).__name__}")
    
    # Handle empty list efficiently
    if not items:
        return [], []
    
    # Partition items into two lists based on predicate
    true_items = []
    false_items = []
    for item in items:
        if predicate(item):
            true_items.append(item)
        else:
            false_items.append(item)
    return true_items, false_items


# ════════════════════════════════════════════════════════════════════════════════
# UNIQUENESS
# ════════════════════════════════════════════════════════════════════════════════

def unique_list(
    items: List[T],
    key_func: Optional[Callable[[T], Any]] = None,
    preserve_order: bool = True
) -> List[T]:
    """
    Get unique items from list.
    
    Args:
        items: List of items
        key_func: Optional function to extract key for uniqueness
        preserve_order: Whether to preserve order of first occurrence
    
    Returns:
        List of unique items
    
    Example:
        >>> unique_list([1, 2, 2, 3, 3, 3])
        [1, 2, 3]
        >>> unique_list([{"id": 1}, {"id": 2}, {"id": 1}], key_func=lambda x: x["id"])
        [{'id': 1}, {'id': 2}]
    """
    if not items:
        return []
    
    if key_func is None:
        if preserve_order:
            seen = set()
            result = []
            for item in items:
                if item not in seen:
                    seen.add(item)
                    result.append(item)
            return result
        else:
            return list(set(items))
    else:
        seen = set()
        result = []
        for item in items:
            key = key_func(item)
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result


# ════════════════════════════════════════════════════════════════════════════════
# DICTIONARY OPERATIONS
# ════════════════════════════════════════════════════════════════════════════════

def filter_dict(
    d: DictStrAny,
    keys: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None
) -> DictStrAny:
    """
    Filter dictionary by keys (include or exclude specific keys).
    
    Args:
        d: Dictionary to filter (must be a dict)
        keys: Keys to include (if None, include all; cannot be used with exclude_keys)
        exclude_keys: Keys to exclude (cannot be used with keys)
    
    Returns:
        Filtered dictionary (empty dict if d is empty)
    
    Raises:
        TypeError: If d is not a dict
        ValueError: If both keys and exclude_keys are provided
    
    Examples:
        >>> filter_dict({"a": 1, "b": 2, "c": 3}, keys=["a", "b"])
        {'a': 1, 'b': 2}
        >>> filter_dict({"a": 1, "b": 2, "c": 3}, exclude_keys=["c"])
        {'a': 1, 'b': 2}
        >>> filter_dict({})
        {}
    """
    # Validate inputs
    if not isinstance(d, dict):
        raise TypeError(f"d must be a dict, got {type(d).__name__}")
    if keys is not None and exclude_keys is not None:
        raise ValueError("Cannot specify both keys and exclude_keys")
    
    # Handle empty dict efficiently
    if not d:
        return {}
    
    # Filter based on include or exclude mode
    if keys is not None:
        # Include only specified keys (convert to set for O(1) lookup)
        keys_set = set(keys)
        return {k: v for k, v in d.items() if k in keys_set}
    elif exclude_keys is not None:
        # Exclude specified keys (convert to set for O(1) lookup)
        exclude_set = set(exclude_keys)
        return {k: v for k, v in d.items() if k not in exclude_set}
    else:
        # No filtering: return copy
        return d.copy()


def merge_dicts(*dicts: DictStrAny, deep: bool = False) -> DictStrAny:
    """
    Merge multiple dictionaries (later dicts override earlier ones).
    
    Args:
        *dicts: Dictionaries to merge (all must be dicts)
        deep: If True, perform deep merge (recursive for nested dicts)
    
    Returns:
        Merged dictionary (empty dict if no dicts provided)
    
    Raises:
        TypeError: If any argument is not a dict
    
    Examples:
        >>> merge_dicts({"a": 1}, {"b": 2}, {"c": 3})
        {'a': 1, 'b': 2, 'c': 3}
        >>> merge_dicts({"a": {"x": 1}}, {"a": {"y": 2}}, deep=True)
        {'a': {'x': 1, 'y': 2}}
        >>> merge_dicts()
        {}
    """
    # Handle no arguments
    if not dicts:
        return {}
    
    # Validate all arguments are dicts
    for i, d in enumerate(dicts):
        if not isinstance(d, dict):
            raise TypeError(f"All arguments must be dicts, got {type(d).__name__} at position {i}")
    
    # Start with copy of first dict
    result = dicts[0].copy()
    
    # Merge remaining dicts
    for d in dicts[1:]:
        if deep:
            # Deep merge: recursively merge nested dicts
            for k, v in d.items():
                if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                    result[k] = merge_dicts(result[k], v, deep=True)
                else:
                    result[k] = v
        else:
            # Shallow merge: later values override earlier ones
            result.update(d)
    
    return result


def get_nested_value(
    d: DictStrAny,
    key_path: str,
    default: Any = None,
    separator: str = "."
) -> Any:
    """
    Get nested value from dictionary using dot notation.
    
    Args:
        d: Dictionary (must be a dict)
        key_path: Path to value (e.g., "a.b.c", must be non-empty)
        default: Default value if path doesn't exist
        separator: Separator for path (must be non-empty)
    
    Returns:
        Value at path or default if path doesn't exist
    
    Raises:
        TypeError: If d is not a dict
        ValueError: If key_path or separator is empty
    
    Examples:
        >>> get_nested_value({"a": {"b": {"c": 1}}}, "a.b.c")
        1
        >>> get_nested_value({"a": {"b": {"c": 1}}}, "a.b.d", default=0)
        0
    """
    # Validate inputs
    if not isinstance(d, dict):
        raise TypeError(f"d must be a dict, got {type(d).__name__}")
    if not key_path:
        raise ValueError("key_path cannot be empty")
    if not separator:
        raise ValueError("separator cannot be empty")
    
    # Handle empty dict
    if not d:
        return default
    
    # Traverse path
    keys = key_path.split(separator)
    current = d
    
    for key in keys:
        if not key:  # Skip empty keys (e.g., from double separators)
            continue
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current


def set_nested_value(
    d: DictStrAny,
    key_path: str,
    value: Any,
    separator: str = "."
) -> None:
    """
    Set nested value in dictionary using dot notation (creates intermediate dicts if needed).
    
    Args:
        d: Dictionary (must be a dict)
        key_path: Path to value (e.g., "a.b.c", must be non-empty)
        value: Value to set
        separator: Separator for path (must be non-empty)
    
    Raises:
        TypeError: If d is not a dict
        ValueError: If key_path or separator is empty
    
    Examples:
        >>> d = {}
        >>> set_nested_value(d, "a.b.c", 1)
        >>> d
        {'a': {'b': {'c': 1}}}
        >>> set_nested_value(d, "a.b.d", 2)
        >>> d["a"]["b"]
        {'c': 1, 'd': 2}
    """
    # Validate inputs
    if not isinstance(d, dict):
        raise TypeError(f"d must be a dict, got {type(d).__name__}")
    if not key_path:
        raise ValueError("key_path cannot be empty")
    if not separator:
        raise ValueError("separator cannot be empty")
    
    # Split path and traverse/create intermediate dicts
    keys = key_path.split(separator)
    current = d
    
    # Navigate to parent of target key, creating dicts as needed
    for key in keys[:-1]:
        if not key:  # Skip empty keys
            continue
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    
    # Set the final value
    final_key = keys[-1]
    if final_key:  # Only set if final key is non-empty
        current[final_key] = value


# ════════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Chunking
    "chunk_list",
    "chunk_list_iter",
    # Flattening
    "flatten_list",
    "flatten_list_deep",
    # Grouping
    "group_by",
    "group_by_key",
    # Partitioning
    "partition",
    # Uniqueness
    "unique_list",
    # Dictionary operations
    "filter_dict",
    "merge_dicts",
    "get_nested_value",
    "set_nested_value",
]




