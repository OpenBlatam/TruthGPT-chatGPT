# Policy Gradients Specification

## Overview
Policy gradient algorithms are fundamental to RLHF, using recently generated samples to update models. This specification outlines a modular implementation that integrates with the rejection sampling pipeline.

## Core Components

### 1. Base Components
| Component          | Description                                                    | Modular Feature                    |
|-------------------|----------------------------------------------------------------|------------------------------------|
| `BasePolicy`      | Abstract base for all policy implementations                  | Unified policy interface           |
| `BaseValue`       | Abstract base for value function implementations              | Pluggable value functions          |
| `BaseAdvantage`   | Abstract base for advantage estimation methods                | Swappable advantage estimators     |

### 2. Policy Components
| Component          | Description                                                    | Implementation Details             |
|-------------------|----------------------------------------------------------------|------------------------------------|
| `VanillaPolicy`   | Implements vanilla policy gradient                            | Basic policy gradient updates      |
| `PPOPolicy`       | Implements PPO algorithm                                      | Proximal policy optimization       |
| `GRPOPolicy`      | Implements GRPO algorithm                                     | Group relative policy optimization |

### 3. Value Components
| Component          | Description                                                    | Implementation Details             |
|-------------------|----------------------------------------------------------------|------------------------------------|
| `ValueEstimator`  | Estimates state values                                        | Value function computation         |
| `GAEEstimator`    | Implements GAE for advantage estimation                        | Generalized advantage estimation   |
| `BaselineEstimator`| Computes baselines for variance reduction                      | Baseline computation               |

## Implementation Details

### Base Component Implementation
```python
class BasePolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def compute_loss(self,
                    logits: torch.Tensor,
                    actions: torch.Tensor,
                    advantages: torch.Tensor) -> torch.Tensor:
        """Compute policy loss."""
        raise NotImplementedError
        
    def update(self, loss: torch.Tensor) -> None:
        """Update policy parameters."""
        raise NotImplementedError

class BaseValue(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def estimate_value(self, states: torch.Tensor) -> torch.Tensor:
        """Estimate state values."""
        raise NotImplementedError
        
    def compute_value_loss(self,
                          values: torch.Tensor,
                          returns: torch.Tensor) -> torch.Tensor:
        """Compute value function loss."""
        raise NotImplementedError
```

### Policy Implementation
```python
class PPOPolicy(BasePolicy):
    def __init__(self, config):
        super().__init__(config)
        self.epsilon = config.epsilon
        self.kl_target = config.kl_target
        
    def compute_loss(self,
                    logits: torch.Tensor,
                    actions: torch.Tensor,
                    advantages: torch.Tensor,
                    old_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute PPO loss with clipping.
        
        Args:
            logits: Current policy logits
            actions: Selected actions
            advantages: Computed advantages
            old_logits: Logits from old policy
            
        Returns:
            PPO loss tensor
        """
        # Compute probability ratios
        new_probs = F.softmax(logits, dim=-1)
        old_probs = F.softmax(old_logits, dim=-1)
        ratios = new_probs / old_probs
        
        # Compute clipped objective
        clipped_ratios = torch.clamp(
            ratios,
            1 - self.epsilon,
            1 + self.epsilon
        )
        
        # Compute losses
        policy_loss = -torch.min(
            ratios * advantages,
            clipped_ratios * advantages
        ).mean()
        
        # Add KL penalty
        kl_div = F.kl_div(
            new_probs.log(),
            old_probs,
            reduction='batchmean'
        )
        kl_penalty = self.kl_target * kl_div
        
        return policy_loss + kl_penalty

class GRPOPolicy(BasePolicy):
    def __init__(self, config):
        super().__init__(config)
        self.epsilon_low = config.epsilon_low
        self.epsilon_high = config.epsilon_high
        
    def compute_loss(self,
                    logits: torch.Tensor,
                    actions: torch.Tensor,
                    advantages: torch.Tensor,
                    old_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute GRPO loss with asymmetric clipping.
        
        Args:
            logits: Current policy logits
            actions: Selected actions
            advantages: Computed advantages
            old_logits: Logits from old policy
            
        Returns:
            GRPO loss tensor
        """
        # Compute probability ratios
        new_probs = F.softmax(logits, dim=-1)
        old_probs = F.softmax(old_logits, dim=-1)
        ratios = new_probs / old_probs
        
        # Compute asymmetric clipping
        clipped_ratios = torch.where(
            advantages > 0,
            torch.clamp(ratios, 1 - self.epsilon_low, 1 + self.epsilon_high),
            torch.clamp(ratios, 1 - self.epsilon_high, 1 + self.epsilon_low)
        )
        
        # Compute loss
        policy_loss = -torch.min(
            ratios * advantages,
            clipped_ratios * advantages
        ).mean()
        
        return policy_loss
```

### Value Implementation
```python
class GAEEstimator(BaseValue):
    def __init__(self, config):
        super().__init__(config)
        self.gamma = config.gamma
        self.lambda_ = config.lambda_
        
    def compute_advantages(self,
                          rewards: torch.Tensor,
                          values: torch.Tensor,
                          next_values: torch.Tensor) -> torch.Tensor:
        """
        Compute GAE advantages.
        
        Args:
            rewards: Reward sequence
            values: Value estimates
            next_values: Next state value estimates
            
        Returns:
            Advantage tensor
        """
        deltas = rewards + self.gamma * next_values - values
        advantages = []
        advantage = 0
        
        for delta in reversed(deltas):
            advantage = delta + self.gamma * self.lambda_ * advantage
            advantages.insert(0, advantage)
            
        return torch.tensor(advantages)
```

## Configuration Parameters

### Policy Parameters
| Parameter     | Type    | Description                                      |
|--------------|---------|--------------------------------------------------|
| `epsilon`    | `float` | PPO clipping parameter (default: `0.2`)          |
| `kl_target`  | `float` | Target KL divergence (default: `0.01`)           |
| `epsilon_low`| `float` | GRPO lower clipping (default: `0.1`)             |
| `epsilon_high`| `float`| GRPO upper clipping (default: `0.3`)             |

### Value Parameters
| Parameter     | Type    | Description                                      |
|--------------|---------|--------------------------------------------------|
| `gamma`      | `float` | Discount factor (default: `0.99`)                |
| `lambda_`    | `float` | GAE parameter (default: `0.95`)                  |
| `value_coef` | `float` | Value loss coefficient (default: `0.5`)          |

## Integration with Rejection Sampling

### Pipeline Components
| Component          | Description                                                    | Integration Point                 |
|-------------------|----------------------------------------------------------------|------------------------------------|
| `RSRewardModel`   | Reward model for rejection sampling                            | Used for initial policy training   |
| `PolicyTrainer`   | Trains policy using selected completions                       | Uses RS outputs for training       |
| `ValueTrainer`    | Trains value function on RS data                               | Uses RS trajectories              |

### Usage Example
```python
# Initialize components
rs_config = RejectionSamplingConfig(
    temperature=0.7,
    top_k=50,
    num_samples=10
)

pg_config = PolicyGradientsConfig(
    epsilon=0.2,
    gamma=0.99,
    lambda_=0.95
)

# Create modular pipeline
rs_pipeline = RejectionSamplingPipeline(rs_config)
policy = PPOPolicy(pg_config)
value = GAEEstimator(pg_config)

# Generate and select completions
completions = rs_pipeline.generate_and_select(prompts)

# Train policy and value function
for batch in completions:
    # Compute advantages
    values = value.estimate_value(batch.states)
    advantages = value.compute_advantages(
        batch.rewards,
        values,
        batch.next_values
    )
    
    # Update policy
    policy_loss = policy.compute_loss(
        batch.logits,
        batch.actions,
        advantages,
        batch.old_logits
    )
    policy.update(policy_loss)
    
    # Update value function
    value_loss = value.compute_value_loss(values, batch.returns)
    value.update(value_loss)
```

## Monitoring and Validation

### Key Metrics
| Metric           | Description                                      |
|-----------------|--------------------------------------------------|
| `policy_loss`   | Policy gradient loss                            |
| `value_loss`    | Value function loss                             |
| `kl_divergence` | KL divergence between old and new policies      |
| `advantage_mean`| Mean of advantage estimates                     |

### Validation Process
1. Policy performance evaluation
2. Value function accuracy
3. Advantage estimation quality
4. KL divergence monitoring 