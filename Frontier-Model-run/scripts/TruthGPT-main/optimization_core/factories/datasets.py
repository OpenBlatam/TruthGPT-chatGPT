from typing import Any, Dict, Iterable, Iterator, Optional

from optimization_core.factories.registry import Registry

DATASETS = Registry()


@DATASETS.register("hf")
def build_hf(dataset: str, subset: Optional[str], text_field: str, streaming: bool = False, limit: Optional[int] = None):
    from datasets import load_dataset

    ds = load_dataset(dataset, subset) if subset else load_dataset(dataset)
    if streaming:
        train = ds["train"].to_iterable_dataset()
        val = ds["validation"].to_iterable_dataset()
        return (
            (ex[text_field] for ex in train.take(limit) if text_field in ex) if limit else (ex[text_field] for ex in train if text_field in ex),
            (ex[text_field] for ex in val.take(max(256, (limit or 0) // 10)) if text_field in ex) if limit else (ex[text_field] for ex in val if text_field in ex),
        )
    train = list(ds["train"][text_field][: limit or len(ds["train"])])
    val_lim = max(256, (limit or len(ds["validation"])) // 10)
    val = list(ds["validation"][text_field][: val_lim])
    return train, val


@DATASETS.register("jsonl")
def build_jsonl(path: str, text_field: str, limit: Optional[int] = None):
    import json

    def reader(p: str, lim: Optional[int]) -> Iterator[str]:
        count = 0
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if text_field in obj:
                        yield obj[text_field]
                        count += 1
                        if lim and count >= lim:
                            break
                except Exception:
                    continue
    # single-file jsonl for both train/val; split by ratio
    all_texts = list(reader(path, limit))
    split = max(1, int(len(all_texts) * 0.9))
    return all_texts[:split], all_texts[split:]


@DATASETS.register("webdataset")
def build_webdataset(url_or_path: str, text_field: str, limit: Optional[int] = None):
    # Placeholder: return empty lists or raise to implement later
    return [], []




