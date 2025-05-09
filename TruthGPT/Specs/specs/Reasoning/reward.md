# Reward Model Specification

## Mathematical Notation


//

Let $X$ be the space of possible queries (e.g., user prompts). For each query $x \in X$, we collect one or more candidate responses $\{y_j\}_{j=1}^{m_x}$ where $m_x$ is the number of candidate responses for query $x$.

The dataset $D$ is defined as:
$$D = \{(x_i, \{y_{ij}\}_{j=1}^{m_i}, \{\text{preferences}_i\})\}_{i=1}^N$$

// 

## Architecture Overview
The reward model is a neural network that learns to predict human preferences by comparing pairs of model outputs. It uses a transformer-based architecture with a specialized reward head for preference prediction.

## Model Parameters

### Core Parameters
| Parameter        | Type    | Description                                      |
|-----------------|---------|--------------------------------------------------|
| `input_dim`     | `int`   | Dimensionality of input features                 |
| `hidden_dim`    | `int`   | Size of hidden layers (default: 4× input_dim)    |
| `dropout`       | `float` | Dropout probability (default: `0.1`)             |
| `num_layers`    | `int`   | Number of transformer layers (default: `6`)      |
| `num_heads`     | `int`   | Number of attention heads (default: `8`)         |

### Training Parameters
| Parameter           | Type    | Description                                      |
|--------------------|---------|--------------------------------------------------|
| `learning_rate`    | `float` | Initial learning rate (default: `1e-4`)          |
| `batch_size`       | `int`   | Training batch size (default: `32`)              |
| `preference_margin`| `float` | Margin for preference loss (default: `0.1`)      |
| `warmup_steps`     | `int`   | Learning rate warmup steps (default: `1000`)     |
| `max_epochs`       | `int`   | Maximum training epochs (default: `1`)           |

### Reward Head Parameters
| Parameter        | Type    | Description                                      |
|-----------------|---------|--------------------------------------------------|
| `head_dim`      | `int`   | Dimension of reward head (default: `256`)        |
| `head_layers`   | `int`   | Number of reward head layers (default: `2`)      |
| `activation`    | `str`   | Activation function (default: `'relu'`)          |

## Input/Output Specifications

### Input Shapes
- Feature Input: `(batch_size, seq_len, input_dim)`
- Preference Pairs: `(batch_size, 2, seq_len, input_dim)`

### Output Shapes
- Reward Scores: `(batch_size, 1)`
- Preference Probabilities: `(batch_size, 1)`

## Training Process

### Loss Function
The model uses a preference-based loss function:
```python
loss = -log(sigmoid(reward_chosen - reward_rejected - margin))
```

### Training Steps
1. Forward pass through transformer layers
2. Compute reward scores for chosen and rejected outputs
3. Calculate preference loss
4. Backpropagate and update weights
5. Apply gradient clipping and learning rate scheduling

## Evaluation Metrics

### Primary Metrics
| Metric           | Description                                      |
|-----------------|--------------------------------------------------|
| `accuracy`      | Binary classification accuracy                   |
| `auc`           | Area under ROC curve                            |
| `reward_diff`   | Mean absolute difference in rewards             |
| `consistency`   | Reward consistency across similar inputs        |

### Validation Process
1. Compute rewards for test pairs
2. Calculate preference accuracy
3. Measure reward distribution statistics
4. Evaluate consistency across similar inputs

## Implementation Notes

### Key Features
- Transformer-based architecture for sequence processing
- Preference-based training with margin
- Gradient clipping for stability
- Learning rate warmup and scheduling
- Dropout for regularization

### Best Practices
1. Use gradient clipping (max norm: 1.0)
2. Apply learning rate warmup
3. Monitor reward distribution
4. Regular validation on held-out pairs
5. Early stopping based on validation metrics

## Usage Example
```python
reward_model = RewardModel(
    input_dim=512,
    hidden_dim=2048,
    dropout=0.1,
    num_layers=6,
    num_heads=8
)

# Training
trainer = RewardModelTrainer(
    model=reward_model,
    learning_rate=1e-4,
    batch_size=32,
    preference_margin=0.1
)

# Evaluation
metrics = trainer.evaluate(
    test_pairs=test_data,
    metrics=['accuracy', 'auc', 'reward_diff']
)
```