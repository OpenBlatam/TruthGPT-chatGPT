# Deprecated Optimizer Files

## ⚠️ Removal Notice

The following optimizer files have been **removed** and are no longer available in the root directory:

- `advanced_truthgpt_optimizer.py`
- `expert_truthgpt_optimizer.py`
- `ultimate_truthgpt_optimizer.py`
- `supreme_truthgpt_optimizer.py`
- `enterprise_truthgpt_optimizer.py`
- `ultra_fast_truthgpt_optimizer.py`
- `ultra_speed_truthgpt_optimizer.py`
- `hyper_speed_truthgpt_optimizer.py`
- `lightning_speed_truthgpt_optimizer.py`
- `infinite_truthgpt_optimizer.py`
- `transcendent_truthgpt_optimizer.py`

## Migration

Please migrate to the unified optimizer system:

```python
# Recommended
from optimizers import create_truthgpt_optimizer
optimizer = create_truthgpt_optimizer('advanced', config)
```

If you need backward compatibility, use the `optimizers.compatibility` module:

```python
# Deprecated (but available)
from optimizers.compatibility import AdvancedTruthGPTOptimizer
optimizer = AdvancedTruthGPTOptimizer(config)  # Shows deprecation warning
```

**Note:** Importing directly from the root package (e.g. `from advanced_truthgpt_optimizer import ...`) will now fail as the files have been removed.

See `REFACTORING_OPTIMIZERS.md` for complete migration guide.







