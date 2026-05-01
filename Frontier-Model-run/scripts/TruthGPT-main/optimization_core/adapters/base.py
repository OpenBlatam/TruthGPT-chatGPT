"""
Base Dynamic Adapter Implementation — Pydantic-First Architecture.

This module provides the BaseDynamicAdapter class, which bridges traditional
procedural adapters into the autonomous ToolRegistry ecosystem (BaseTool).
This allows agents to dynamically discover and compose adapter operations
(Input -> Process -> Output) without hardcoded sequences.

The **ObjectStore** is a thread-safe, process-global registry that allows
adapters to stash heavyweight Python objects (models, datasets, optimizers)
and return lightweight string IDs to the AI agent.
"""

import logging
import json
import time
import threading
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, computed_field

try:
    from optimization_core.agents.razonamiento_planificacion.tools import BaseTool, ToolResult
except (ImportError, ValueError, KeyError):
    # Fallback for environments where optimization_core is the root
    from agents.razonamiento_planificacion.tools import BaseTool, ToolResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class ObjectEntry(BaseModel):
    """Typed metadata for an object stored in the ObjectStore."""
    obj_id: str
    kind: str = "unknown"
    meta: Dict[str, Any] = Field(default_factory=dict)
    stored_at: float = Field(default_factory=time.time)

    @computed_field  # type: ignore[misc]
    @property
    def age_seconds(self) -> float:
        return round(time.time() - self.stored_at, 2)


class StoreStats(BaseModel):
    """Snapshot of the ObjectStore state."""
    total_objects: int = 0
    kinds: Dict[str, int] = Field(default_factory=dict)


class AdapterRunResult(BaseModel):
    """Structured metadata from a BaseDynamicAdapter execution."""
    adapter_name: str
    status: str  # "success" | "error"
    elapsed_ms: float = 0.0
    input_keys: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Process-global Object Store
# ---------------------------------------------------------------------------

class ObjectStore:
    """
    Thread-safe, in-memory object store for heavyweight Python objects.

    Adapters ``store()`` raw objects (DataLoaders, torch.nn.Module, Optimizers)
    and receive a lightweight string ID.  Other adapters can ``get()`` the
    object by ID, enabling JSON-only agent communication while still passing
    real Python references internally.

    Usage::

        store = ObjectStore.instance()
        obj_id = store.put(my_dataloader, kind="dataset", meta={"rows": 50_000})
        loader = store.get(obj_id)
        store.delete(obj_id)
    """

    _singleton: Optional["ObjectStore"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._objects: Dict[str, Any] = {}
        self._entries: Dict[str, ObjectEntry] = {}
        self._obj_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "ObjectStore":
        """Return the process-global singleton."""
        if cls._singleton is None:
            with cls._lock:
                if cls._singleton is None:
                    cls._singleton = cls()
        return cls._singleton

    # -- Public API ---------------------------------------------------------

    def put(self, obj: Any, *, kind: str = "unknown", meta: Optional[Dict[str, Any]] = None) -> str:
        """Store *obj* and return a unique string ID."""
        obj_id = f"{kind}_{uuid.uuid4().hex[:12]}"
        entry = ObjectEntry(obj_id=obj_id, kind=kind, meta=meta or {})
        with self._obj_lock:
            self._objects[obj_id] = obj
            self._entries[obj_id] = entry
        logger.info("ObjectStore: stored %s (kind=%s)", obj_id, kind)
        return obj_id

    def get(self, obj_id: str) -> Any:
        """Retrieve the raw object by *obj_id*.  Raises ``KeyError`` if not found."""
        with self._obj_lock:
            obj = self._objects.get(obj_id)
        if obj is None:
            raise KeyError(f"ObjectStore: ID '{obj_id}' not found. Available: {list(self._objects.keys())}")
        return obj

    def get_entry(self, obj_id: str) -> ObjectEntry:
        """Return the typed ``ObjectEntry`` for *obj_id*."""
        with self._obj_lock:
            entry = self._entries.get(obj_id)
        if entry is None:
            raise KeyError(f"ObjectStore: ID '{obj_id}' not found.")
        return entry

    def get_meta(self, obj_id: str) -> Dict[str, Any]:
        """Return the metadata dict attached to *obj_id*."""
        return self.get_entry(obj_id).meta

    def delete(self, obj_id: str) -> bool:
        """Remove *obj_id* from the store.  Returns True if deleted."""
        with self._obj_lock:
            removed_obj = self._objects.pop(obj_id, None)
            self._entries.pop(obj_id, None)
        if removed_obj is not None:
            logger.info("ObjectStore: deleted %s", obj_id)
            return True
        return False

    def list_ids(self, kind: Optional[str] = None) -> List[str]:
        """Return all stored IDs, optionally filtered by *kind*."""
        with self._obj_lock:
            if kind:
                return [k for k, e in self._entries.items() if e.kind == kind]
            return list(self._entries.keys())

    def list_entries(self, kind: Optional[str] = None) -> List[ObjectEntry]:
        """Return typed entries, optionally filtered by *kind*."""
        with self._obj_lock:
            entries = list(self._entries.values())
        if kind:
            entries = [e for e in entries if e.kind == kind]
        return entries

    def stats(self) -> StoreStats:
        """Return a typed snapshot of store statistics."""
        with self._obj_lock:
            kinds: Dict[str, int] = {}
            for e in self._entries.values():
                kinds[e.kind] = kinds.get(e.kind, 0) + 1
            return StoreStats(total_objects=len(self._entries), kinds=kinds)

    def clear(self) -> int:
        """Remove all objects and return the count of items cleared."""
        with self._obj_lock:
            count = len(self._objects)
            self._objects.clear()
            self._entries.clear()
        logger.info("ObjectStore: cleared %d objects", count)
        return count


# ---------------------------------------------------------------------------
# Base Dynamic Adapter
# ---------------------------------------------------------------------------

class BaseDynamicAdapter(BaseTool):
    """
    Base class for all intelligence-driven adapters.

    Inherits from BaseTool to be directly discoverable by the Agent Swarm.
    Children must define ``name``, ``description``, and ``process()``.
    """

    @property
    def store(self) -> ObjectStore:
        """Convenience accessor to the global ObjectStore singleton."""
        return ObjectStore.instance()

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        The core logic of the adapter.
        Input -> Process -> Output
        """

    async def run(self, tool_input: str) -> ToolResult:
        """
        Tool execution entry point.  Agents pass stringified JSON inputs.
        Provides a Security & Validation layer before reaching core logic.
        """
        start = time.monotonic()
        try:
            input_data = json.loads(tool_input)
            input_keys = list(input_data.keys())
            logger.info("Adapter '%s' executing with input keys: %s", self.name, input_keys)

            output_data = self.process(input_data)

            elapsed_ms = round((time.monotonic() - start) * 1000, 2)
            result_str = json.dumps(output_data, indent=2, default=str)
            run_result = AdapterRunResult(
                adapter_name=self.name,
                status="success",
                elapsed_ms=elapsed_ms,
                input_keys=input_keys,
            )
            return ToolResult(
                output=result_str,
                metadata=run_result.model_dump(),
            )
        except json.JSONDecodeError as exc:
            elapsed_ms = round((time.monotonic() - start) * 1000, 2)
            logger.error("Adapter '%s' failed to parse JSON input: %s", self.name, exc)
            return ToolResult(
                output=f"Error: Invalid JSON input provided to {self.name}. Details: {exc}",
                metadata=AdapterRunResult(
                    adapter_name=self.name, status="error", elapsed_ms=elapsed_ms,
                ).model_dump(),
            )
        except Exception as exc:
            elapsed_ms = round((time.monotonic() - start) * 1000, 2)
            logger.exception("Adapter '%s' execution failed", self.name)
            return ToolResult(
                output=f"Error during {self.name} execution: {exc}",
                metadata=AdapterRunResult(
                    adapter_name=self.name, status="error", elapsed_ms=elapsed_ms,
                ).model_dump(),
            )


