# Regularization Specification

## Overview
Regularization in RLHF is crucial for preventing over-optimization of reward models and maintaining model behavior stability. This specification outlines the key regularization techniques and their implementations.

## Core Components

### 1. KL Distance Regularization
| Component          | Description                                                    | Implementation Details                    |
|-------------------|----------------------------------------------------------------|------------------------------------------|
| `KLController`    | Manages KL divergence between policy and reference model      | Implements eq. 2 from RLHF book          |
| `ReferenceModel`  | Static reference model for KL calculations                     | Usually instruction-tuned model          |
| `KLCalculator`    | Computes KL divergence between distributions                  | Implements eq. 3 from RLHF book          |

### 2. Pretraining Gradient Regularization
| Component          | Description                                                    | Implementation Details                    |
|-------------------|----------------------------------------------------------------|------------------------------------------|
| `PretrainReg`     | Maintains performance on pretraining tasks                     | Implements eq. 6 from RLHF book          |
| `GradientBalancer`| Balances RL and pretraining gradients                          | Weighted combination of gradients        |
| `DatasetMonitor`  | Tracks performance on pretraining datasets                     | Metrics collection and monitoring        |

### 3. Reward Model Regularization
| Component          | Description                                                    | Implementation Details                    |
|-------------------|----------------------------------------------------------------|------------------------------------------|
| `MarginLoss`      | Implements margin-based preference loss                        | Implements eq. 9 from RLHF book          |
| `RewardValidator` | Validates reward model predictions                            | Consistency and distribution checks      |
| `PreferenceReg`   | Regularizes preference predictions                            | Contrastive loss with margins            |

## Implementation Details

### KL Distance Implementation
```python
class KLController:
    def __init__(self, lambda_kl: float = 0.1):
        self.lambda_kl = lambda_kl
        
    def compute_kl_penalty(self, 
                          policy_logits: torch.Tensor,
                          ref_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence penalty between policy and reference model.
        
        Args:
            policy_logits: Logits from current policy
            ref_logits: Logits from reference model
            
        Returns:
            KL penalty tensor
        """
        policy_logprobs = F.log_softmax(policy_logits, dim=-1)
        ref_logprobs = F.log_softmax(ref_logits, dim=-1)
        
        # Compute KL divergence
        kl_div = F.kl_div(
            policy_logprobs,
            ref_logprobs,
            reduction='batchmean'
        )
        
        return self.lambda_kl * kl_div
```

### Pretraining Regularization
```python
class PretrainRegularizer:
    def __init__(self, gamma: float = 0.1):
        self.gamma = gamma
        
    def compute_pretrain_loss(self,
                            model_logits: torch.Tensor,
                            pretrain_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute pretraining loss for regularization.
        
        Args:
            model_logits: Model output logits
            pretrain_labels: Labels from pretraining data
            
        Returns:
            Pretraining loss tensor
        """
        nll_loss = F.cross_entropy(
            model_logits.view(-1, model_logits.size(-1)),
            pretrain_labels.view(-1)
        )
        
        return self.gamma * nll_loss
```

### Reward Model Regularization
```python
class RewardRegularizer:
    def __init__(self, margin: float = 0.1):
        self.margin = margin
        
    def compute_margin_loss(self,
                          chosen_rewards: torch.Tensor,
                          rejected_rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute margin-based preference loss.
        
        Args:
            chosen_rewards: Rewards for preferred completions
            rejected_rewards: Rewards for rejected completions
            
        Returns:
            Margin loss tensor
        """
        return -torch.log(
            torch.sigmoid(chosen_rewards - rejected_rewards - self.margin)
        ).mean()
```

## Training Process

### Regularization Pipeline
1. **KL Control**
   - Sample from current policy
   - Compute KL divergence with reference model
   - Apply KL penalty to reward

2. **Pretraining Balance**
   - Compute pretraining loss
   - Balance with RL objective
   - Monitor performance metrics

3. **Reward Regularization**
   - Apply margin-based loss
   - Validate reward distributions
   - Monitor preference consistency

## Configuration Parameters

### KL Regularization
| Parameter     | Type    | Description                                      |
|--------------|---------|--------------------------------------------------|
| `lambda_kl`  | `float` | KL penalty weight (default: `0.1`)               |
| `ref_model`  | `str`   | Reference model path or identifier               |
| `kl_threshold`| `float`| Maximum allowed KL divergence                    |

### Pretraining Regularization
| Parameter     | Type    | Description                                      |
|--------------|---------|--------------------------------------------------|
| `gamma`      | `float` | Pretraining loss weight (default: `0.1`)         |
| `pretrain_data`| `str`  | Path to pretraining dataset                      |
| `metrics`    | `list`  | Metrics to monitor                               |

### Reward Regularization
| Parameter     | Type    | Description                                      |
|--------------|---------|--------------------------------------------------|
| `margin`     | `float` | Preference margin (default: `0.1`)               |
| `validation_freq`| `int` | How often to validate rewards                    |
| `consistency_threshold`| `float` | Minimum reward consistency score          |

## Usage Example
```python
# Initialize regularization components
kl_controller = KLController(lambda_kl=0.1)
pretrain_reg = PretrainRegularizer(gamma=0.1)
reward_reg = RewardRegularizer(margin=0.1)

# Training loop
for batch in dataloader:
    # Forward pass
    policy_logits = model(batch.inputs)
    ref_logits = ref_model(batch.inputs)
    
    # Compute regularization terms
    kl_penalty = kl_controller.compute_kl_penalty(policy_logits, ref_logits)
    pretrain_loss = pretrain_reg.compute_pretrain_loss(policy_logits, batch.labels)
    reward_loss = reward_reg.compute_margin_loss(chosen_rewards, rejected_rewards)
    
    # Combined loss
    total_loss = reward_loss + kl_penalty + pretrain_loss
    
    # Backward pass
    total_loss.backward()
    optimizer.step()
```

## Monitoring and Validation

### Key Metrics
| Metric           | Description                                      |
|-----------------|--------------------------------------------------|
| `kl_divergence` | KL divergence between policy and reference       |
| `pretrain_acc`  | Accuracy on pretraining tasks                    |
| `reward_consistency` | Consistency of reward predictions           |
| `margin_satisfaction` | Percentage of satisfied margins            |

### Validation Process
1. Regular KL divergence checks
2. Pretraining performance monitoring
3. Reward distribution analysis
4. Preference consistency validation 