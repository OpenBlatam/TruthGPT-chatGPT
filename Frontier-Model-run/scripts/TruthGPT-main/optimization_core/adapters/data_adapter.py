"""
Data adapters — Pydantic-First Architecture.

The ``process()`` method performs *real* data loading and stores the result
in the global ObjectStore, returning typed Pydantic results with a
lightweight ``data_id`` that other adapters can consume via JSON.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any

from pydantic import BaseModel, Field, computed_field

from .base import BaseDynamicAdapter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Response Models
# ---------------------------------------------------------------------------

class DataSplitStats(BaseModel):
    """Summary statistics for a data split."""
    num_samples: int = 0
    avg_word_length: float = 0.0


class DataLoadResult(BaseModel):
    """Typed result from a data load action."""
    status: str = "success"
    data_id: str
    train_samples: int
    val_samples: int

    @computed_field  # type: ignore[misc]
    @property
    def total_samples(self) -> int:
        return self.train_samples + self.val_samples


class DataInfoResult(BaseModel):
    """Typed result from a data info action."""
    status: str = "success"
    data_id: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class DataListResult(BaseModel):
    """Typed result from a data list action."""
    status: str = "success"
    datasets: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Core Adapter
# ---------------------------------------------------------------------------

class DataAdapter(BaseDynamicAdapter):
    """Base adapter for data operations."""

    name: str = "data_adapter"
    description: str = (
        "Adapter to load and analyze datasets. Input JSON: "
        "{'action': 'load'|'info'|'list', 'source': 'str', 'kwargs': {}}"
    )

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        action = input_data.get("action")
        kwargs = input_data.get("kwargs", {})

        if action == "load":
            source = input_data.get("source", "")
            train_data, val_data = self.load_data(source, **kwargs)
            data_id = self.store.put(
                {"train": train_data, "val": val_data},
                kind="dataset",
                meta={"source": source, "train_samples": len(train_data), "val_samples": len(val_data)},
            )
            return DataLoadResult(
                data_id=data_id,
                train_samples=len(train_data),
                val_samples=len(val_data),
            ).model_dump()

        elif action == "info":
            data_id = input_data.get("data_id", "")
            if data_id:
                meta = self.store.get_meta(data_id)
                return DataInfoResult(data_id=data_id, meta=meta).model_dump()
            return {"status": "success", "message": "Pass a data_id to retrieve info."}

        elif action == "list":
            ids = self.store.list_ids(kind="dataset")
            return DataListResult(datasets=ids).model_dump()

        else:
            raise ValueError(f"Unknown data action: '{action}'. Use 'load', 'info', or 'list'.")

    def load_data(self, source: str, **kwargs) -> Tuple[List[str], List[str]]:
        """Load training and validation data.  Override in subclasses."""
        raise NotImplementedError("Subclass must implement load_data()")

    def get_data_info(self, data: List[str]) -> DataSplitStats:
        """Get typed summary statistics about a data split."""
        avg_len = sum(len(text.split()) for text in data) / max(1, len(data))
        return DataSplitStats(num_samples=len(data), avg_word_length=round(avg_len, 2))


class HuggingFaceDataAdapter(DataAdapter):
    """Adapter for HuggingFace datasets."""

    name: str = "hf_data_adapter"
    description: str = (
        "Load datasets from HuggingFace Hub. Input JSON: "
        "{'action': 'load', 'source': 'dataset_name', 'kwargs': {'subset': 'optional', 'text_field': 'text'}}"
    )

    def load_data(self, source: str, **kwargs) -> Tuple[List[str], List[str]]:
        """Load data from HuggingFace."""
        from datasets import load_dataset

        dataset_name = kwargs.get("dataset_name", source)
        subset = kwargs.get("subset")
        text_field = kwargs.get("text_field", "text")
        streaming = kwargs.get("streaming", False)

        if subset:
            ds = load_dataset(dataset_name, subset, streaming=streaming)
        else:
            ds = load_dataset(dataset_name, streaming=streaming)

        train_data = ds["train"][text_field] if not streaming else ds["train"]

        if "validation" in ds:
            val_data = ds["validation"][text_field] if not streaming else ds["validation"]
        elif "val" in ds:
            val_data = ds["val"][text_field] if not streaming else ds["val"]
        else:
            val_data = train_data

        train_list = list(train_data) if not isinstance(train_data, list) else train_data
        val_list = list(val_data) if not isinstance(val_data, list) else val_data

        return train_list, val_list


class JSONLDataAdapter(DataAdapter):
    """Adapter for JSONL files."""

    name: str = "jsonl_data_adapter"
    description: str = (
        "Load datasets from local JSONL files. Input JSON: "
        "{'action': 'load', 'source': '/path/to/file.jsonl', 'kwargs': {'text_field': 'text', 'train_split': 0.9}}"
    )

    def load_data(self, source: str, **kwargs) -> Tuple[List[str], List[str]]:
        """Load data from JSONL file."""
        import json as _json

        text_field = kwargs.get("text_field", "text")
        train_split = kwargs.get("train_split", 0.9)

        texts: List[str] = []
        with open(source, "r", encoding="utf-8") as f:
            for line in f:
                data = _json.loads(line.strip())
                if text_field in data:
                    texts.append(data[text_field])

        split_idx = int(len(texts) * train_split)
        return texts[:split_idx], texts[split_idx:]

