# Direct Alignment Algorithms Specification

## Overview
Direct Alignment Algorithms (DAAs) optimize models to solve the RLHF objective without training intermediate reward models or using reinforcement learning optimizers. This specification outlines a modular implementation focusing on Direct Preference Optimization (DPO) and its variants.

## Core Components

### 1. Base Components
| Component          | Description                                                    | Modular Feature                    |
|-------------------|----------------------------------------------------------------|------------------------------------|
| `BaseDAA`         | Abstract base for all DAA implementations                     | Unified DAA interface              |
| `BaseReference`   | Abstract base for reference model implementations             | Pluggable reference models         |
| `BasePreference`  | Abstract base for preference modeling                         | Swappable preference models        |

### 2. DAA Components
| Component          | Description                                                    | Implementation Details             |
|-------------------|----------------------------------------------------------------|------------------------------------|
| `DPO`             | Implements Direct Preference Optimization                      | Core DPO algorithm                 |
| `SLiC`            | Implements Sequence Likelihood Calibration                     | SLiC-HF implementation             |
| `SimPO`           | Implements Simple Preference Optimization                      | Reference-free optimization        |

### 3. Preference Components
| Component          | Description                                                    | Implementation Details             |
|-------------------|----------------------------------------------------------------|------------------------------------|
| `BradleyTerry`    | Implements Bradley-Terry preference model                      | Pairwise preference modeling       |
| `PreferenceLoss`  | Computes preference-based losses                              | Loss computation                   |
| `PreferenceValidator`| Validates preference predictions                             | Preference validation              |

## Implementation Details

### Base Component Implementation
```python
class BaseDAA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.beta = config.beta  # Temperature parameter
        
    def compute_loss(self,
                    policy_logits: torch.Tensor,
                    ref_logits: torch.Tensor,
                    chosen_ids: torch.Tensor,
                    rejected_ids: torch.Tensor) -> torch.Tensor:
        """Compute DAA loss."""
        raise NotImplementedError
        
    def update(self, loss: torch.Tensor) -> None:
        """Update model parameters."""
        raise NotImplementedError

class BaseReference(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def compute_log_probs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities for reference model."""
        raise NotImplementedError
```

### DPO Implementation
```python
class DPO(BaseDAA):
    def __init__(self, config):
        super().__init__(config)
        
    def compute_loss(self,
                    policy_logits: torch.Tensor,
                    ref_logits: torch.Tensor,
                    chosen_ids: torch.Tensor,
                    rejected_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute DPO loss using Bradley-Terry model.
        
        Args:
            policy_logits: Logits from current policy
            ref_logits: Logits from reference model
            chosen_ids: IDs of chosen completions
            rejected_ids: IDs of rejected completions
            
        Returns:
            DPO loss tensor
        """
        # Compute log probabilities
        policy_chosen_logprobs = self._get_logprobs(policy_logits, chosen_ids)
        policy_rejected_logprobs = self._get_logprobs(policy_logits, rejected_ids)
        ref_chosen_logprobs = self._get_logprobs(ref_logits, chosen_ids)
        ref_rejected_logprobs = self._get_logprobs(ref_logits, rejected_ids)
        
        # Compute log ratios
        chosen_logratios = policy_chosen_logprobs - ref_chosen_logprobs
        rejected_logratios = policy_rejected_logprobs - ref_rejected_logprobs
        
        # Compute DPO loss
        losses = -F.logsigmoid(self.beta * (chosen_logratios - rejected_logratios))
        
        return losses.mean()
```

### SLiC Implementation
```python
class SLiC(BaseDAA):
    def __init__(self, config):
        super().__init__(config)
        self.margin = config.margin
        
    def compute_loss(self,
                    policy_logits: torch.Tensor,
                    ref_logits: torch.Tensor,
                    chosen_ids: torch.Tensor,
                    rejected_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute SLiC loss with margin.
        
        Args:
            policy_logits: Logits from current policy
            ref_logits: Logits from reference model
            chosen_ids: IDs of chosen completions
            rejected_ids: IDs of rejected completions
            
        Returns:
            SLiC loss tensor
        """
        # Compute log probabilities
        policy_chosen_logprobs = self._get_logprobs(policy_logits, chosen_ids)
        policy_rejected_logprobs = self._get_logprobs(policy_logits, rejected_ids)
        
        # Compute margin loss
        losses = F.relu(
            self.margin - (policy_chosen_logprobs - policy_rejected_logprobs)
        )
        
        return losses.mean()
```

## Configuration Parameters

### DAA Parameters
| Parameter     | Type    | Description                                      |
|--------------|---------|--------------------------------------------------|
| `beta`       | `float` | Temperature parameter (default: `0.1`)           |
| `margin`     | `float` | Margin for SLiC (default: `0.1`)                 |
| `ref_weight` | `float` | Reference model weight (default: `1.0`)          |

### Preference Parameters
| Parameter     | Type    | Description                                      |
|--------------|---------|--------------------------------------------------|
| `batch_size` | `int`   | Batch size for training (default: `32`)          |
| `max_length` | `int`   | Maximum sequence length (default: `512`)         |
| `warmup_steps`| `int`  | Warmup steps (default: `100`)                    |

## Integration with Existing Components

### Pipeline Components
| Component          | Description                                                    | Integration Point                 |
|-------------------|----------------------------------------------------------------|------------------------------------|
| `PolicyGradients` | Policy gradient algorithms                                     | Alternative optimization method    |
| `RejectionSampling`| Rejection sampling pipeline                                   | Data generation                    |
| `RewardModel`     | Reward model for comparison                                    | Performance baseline               |

### Usage Example
```python
# Initialize components
daa_config = DAAConfig(
    beta=0.1,
    margin=0.1,
    batch_size=32
)

# Create modular pipeline
dpo = DPO(daa_config)
reference = BaseReference(daa_config)
preference = BradleyTerry(daa_config)

# Training loop
for batch in dataloader:
    # Get logits from policy and reference
    policy_logits = model(batch.input_ids)
    ref_logits = reference(batch.input_ids)
    
    # Compute DPO loss
    loss = dpo.compute_loss(
        policy_logits,
        ref_logits,
        batch.chosen_ids,
        batch.rejected_ids
    )
    
    # Update model
    dpo.update(loss)
    
    # Validate preferences
    preference.validate(
        policy_logits,
        batch.chosen_ids,
        batch.rejected_ids
    )
```

## Monitoring and Validation

### Key Metrics
| Metric           | Description                                      |
|-----------------|--------------------------------------------------|
| `daa_loss`      | Direct alignment loss                           |
| `preference_acc`| Preference prediction accuracy                  |
| `kl_divergence` | KL divergence from reference model              |
| `chosen_reward` | Reward for chosen completions                   |

### Validation Process
1. Preference prediction accuracy
2. KL divergence monitoring
3. Chosen vs rejected reward comparison
4. Reference model alignment 