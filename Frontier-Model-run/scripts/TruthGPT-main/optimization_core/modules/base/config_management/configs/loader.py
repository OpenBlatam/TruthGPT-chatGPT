from __future__ import annotations

from typing import Any, Dict, List

import yaml

from .schema import AppCfg


def _set_in(dct: Dict[str, Any], keys: List[str], value: Any) -> None:
    cur = dct
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def parse_overrides(kvs: List[str] | None) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if not kvs:
        return result
    for item in kvs:
        if "=" not in item:
            continue
        key, val = item.split("=", 1)
        _set_in(result, key.split("."), _parse_scalar(val))
    return result


def _parse_scalar(v: str) -> Any:
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    try:
        if "." in v:
            return float(v)
        return int(v)
    except Exception:
        return v


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str, overrides: List[str] | None = None) -> AppCfg:
    base: Dict[str, Any] = yaml.safe_load(open(path, "r", encoding="utf-8"))
    merged = deep_merge(base, parse_overrides(overrides))
    # Validate and coerce
    return AppCfg(**merged)






