import math
from typing import Dict, Any

from optimization_core.factories.registry import Registry

METRICS = Registry()


@METRICS.register("loss")
def metric_loss(context: Dict[str, Any]) -> float:
    # Expects context["val_loss"]
    return float(context.get("val_loss", float("inf")))


@METRICS.register("ppl")
def metric_ppl(context: Dict[str, Any]) -> float:
    val_loss = float(context.get("val_loss", float("inf")))
    try:
        return math.exp(min(20.0, max(-20.0, val_loss)))
    except Exception:
        return float("inf")




