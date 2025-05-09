# Rejection Sampling Specification

## Overview
Rejection Sampling (RS) is a baseline method for preference fine-tuning that curates candidate instructions, filters them using a reward model, and fine-tunes on top completions. This specification outlines a modular implementation.

## Core Components

### 1. Base Components
| Component          | Description                                                    | Modular Feature                    |
|-------------------|----------------------------------------------------------------|------------------------------------|
| `BaseSampler`     | Abstract base for all sampling strategies                      | Unified sampling interface         |
| `BaseSelector`    | Abstract base for completion selection strategies              | Pluggable selection methods        |
| `BaseRewardModel` | Abstract interface for reward model implementations            | Swappable reward models            |

### 2. Sampling Components
| Component          | Description                                                    | Implementation Details             |
|-------------------|----------------------------------------------------------------|------------------------------------|
| `CompletionGenerator`| Generates multiple completions per prompt                     | Temperature and top-k/p sampling   |
| `BatchProcessor`  | Handles batched generation and reward computation              | Efficient batch processing         |
| `LengthOptimizer` | Optimizes batch processing by sorting by length                | Performance optimization           |

### 3. Selection Components
| Component          | Description                                                    | Implementation Details             |
|-------------------|----------------------------------------------------------------|------------------------------------|
| `TopPerPromptSelector`| Selects best completion per prompt                           | Per-prompt optimization           |
| `TopKSelector`    | Selects top K completions across all prompts                   | Global optimization               |
| `HybridSelector`  | Combines multiple selection strategies                         | Flexible selection                |

## Implementation Details

### Base Component Implementation
```python
class BaseSampler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def generate_completions(self, prompts: List[str]) -> List[List[str]]:
        """Generate completions for each prompt."""
        raise NotImplementedError
        
    def compute_rewards(self, 
                       prompts: List[str], 
                       completions: List[List[str]]) -> torch.Tensor:
        """Compute rewards for prompt-completion pairs."""
        raise NotImplementedError

class BaseSelector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def select_completions(self,
                          rewards: torch.Tensor,
                          completions: List[List[str]]) -> List[str]:
        """Select best completions based on rewards."""
        raise NotImplementedError
```

### Sampling Implementation
```python
class CompletionGenerator(BaseSampler):
    def __init__(self, config):
        super().__init__(config)
        self.temperature = config.temperature
        self.top_k = config.top_k
        self.top_p = config.top_p
        
    def generate_completions(self, prompts: List[str]) -> List[List[str]]:
        """
        Generate N completions for each prompt.
        
        Args:
            prompts: List of input prompts
            
        Returns:
            List of lists of completions
        """
        completions = []
        for prompt in prompts:
            # Generate N completions with sampling parameters
            prompt_completions = self._generate_with_params(
                prompt,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p
            )
            completions.append(prompt_completions)
        return completions

class BatchProcessor(BaseSampler):
    def __init__(self, config):
        super().__init__(config)
        self.length_optimizer = LengthOptimizer()
        
    def compute_rewards(self,
                       prompts: List[str],
                       completions: List[List[str]]) -> torch.Tensor:
        """
        Compute rewards for all prompt-completion pairs efficiently.
        
        Args:
            prompts: List of input prompts
            completions: List of lists of completions
            
        Returns:
            Tensor of rewards
        """
        # Sort by length for efficient batching
        sorted_pairs = self.length_optimizer.sort_by_length(prompts, completions)
        
        # Compute rewards in batches
        rewards = []
        for batch in self._create_batches(sorted_pairs):
            batch_rewards = self.reward_model(batch)
            rewards.extend(batch_rewards)
            
        return torch.tensor(rewards)
```

### Selection Implementation
```python
class TopPerPromptSelector(BaseSelector):
    def select_completions(self,
                          rewards: torch.Tensor,
                          completions: List[List[str]]) -> List[str]:
        """
        Select best completion for each prompt.
        
        Args:
            rewards: Tensor of rewards for all completions
            completions: List of lists of completions
            
        Returns:
            List of selected completions
        """
        selected = []
        for i, prompt_completions in enumerate(completions):
            # Get rewards for this prompt's completions
            prompt_rewards = rewards[i * len(prompt_completions):
                                   (i + 1) * len(prompt_completions)]
            
            # Select completion with highest reward
            best_idx = torch.argmax(prompt_rewards)
            selected.append(prompt_completions[best_idx])
            
        return selected

class TopKSelector(BaseSelector):
    def select_completions(self,
                          rewards: torch.Tensor,
                          completions: List[List[str]]) -> List[str]:
        """
        Select top K completions across all prompts.
        
        Args:
            rewards: Tensor of rewards for all completions
            completions: List of lists of completions
            
        Returns:
            List of selected completions
        """
        # Flatten completions and get top K indices
        flat_completions = [c for prompt_completions in completions 
                          for c in prompt_completions]
        top_k_indices = torch.topk(rewards, k=self.config.top_k).indices
        
        return [flat_completions[i] for i in top_k_indices]
```

## Configuration Parameters

### Sampling Parameters
| Parameter     | Type    | Description                                      |
|--------------|---------|--------------------------------------------------|
| `temperature`| `float` | Sampling temperature (default: `0.7`)            |
| `top_k`      | `int`   | Top-k sampling parameter (default: `50`)         |
| `top_p`      | `float` | Top-p sampling parameter (default: `0.9`)        |
| `num_samples`| `int`   | Completions per prompt (default: `10`)           |

### Selection Parameters
| Parameter     | Type    | Description                                      |
|--------------|---------|--------------------------------------------------|
| `selection_method`| `str` | Selection strategy (default: `"top_per_prompt"`) |
| `top_k`      | `int`   | Number of top completions to select              |
| `reward_threshold`| `float`| Minimum reward threshold for selection           |

## Usage Example
```python
# Initialize components
config = RejectionSamplingConfig(
    temperature=0.7,
    top_k=50,
    num_samples=10,
    selection_method="top_per_prompt"
)

# Create modular pipeline
generator = CompletionGenerator(config)
processor = BatchProcessor(config)
selector = TopPerPromptSelector(config)

# Generate and select completions
completions = generator.generate_completions(prompts)
rewards = processor.compute_rewards(prompts, completions)
selected = selector.select_completions(rewards, completions)

# Fine-tune on selected completions
trainer = InstructionTrainer(model, tokenizer)
trainer.train(selected)
```

## Monitoring and Validation

### Key Metrics
| Metric           | Description                                      |
|-----------------|--------------------------------------------------|
| `reward_stats`  | Statistics of reward distribution                |
| `selection_ratio`| Ratio of selected to total completions          |
| `diversity_score`| Diversity of selected completions               |
| `quality_score` | Quality assessment of selected completions       |

### Validation Process
1. Reward distribution analysis
2. Selection strategy evaluation
3. Completion quality assessment
4. Fine-tuning performance monitoring 