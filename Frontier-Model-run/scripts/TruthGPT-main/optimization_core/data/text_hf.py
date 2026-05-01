from typing import Any, Dict, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer

from .registry import register_dataset


def _get_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@register_dataset("hf_text")
def build_hf_text(cfg: Dict[str, Any]) -> Tuple[Any, Any]:
    data_cfg = cfg.get("data", {})
    model_name = cfg["model"]["name_or_path"]
    dataset = str(data_cfg.get("dataset", "wikitext"))
    subset = str(data_cfg.get("subset", "wikitext-2-raw-v1"))
    text_field = str(data_cfg.get("text_field", "text"))
    max_len = int(data_cfg.get("max_seq_len", 512))
    num_proc = int(data_cfg.get("num_proc", 4))

    tok = _get_tokenizer(model_name)
    ds = load_dataset(dataset, subset)

    def encode(batch):
        out = tok(
            batch[text_field],
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )
        out["labels"] = out["input_ids"].copy()
        return out

    ds = ds.map(encode, batched=True, num_proc=num_proc, remove_columns=ds["train"].column_names)
    return ds["train"], ds.get("validation") or ds.get("test")






