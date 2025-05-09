# Instruction Tuning Specification

## Overview
Instruction tuning (IFT) is the foundation of post-training and creating helpful language models. It adapts language models to follow instructions and serves as the base for RLHF.

## Core Components

### 1. Chat Template System
| Component          | Description                                                    | Implementation Details                    |
|-------------------|----------------------------------------------------------------|------------------------------------------|
| `ChatTemplate`    | Formats user queries for model processing                      | Jinja2 template system                   |
| `MessageFormatter`| Handles message role formatting                                | Role-based message structure             |
| `TokenProcessor`  | Manages special tokens and boundaries                          | Token sequence processing                |

### 2. Message Roles
| Role        | Description                                                    | Usage                                    |
|------------|----------------------------------------------------------------|------------------------------------------|
| `system`   | Initial instructions for the model                            | First message only                       |
| `user`     | Input from the user                                           | Alternating with assistant               |
| `assistant`| Model's responses                                             | Alternating with user                    |

### 3. Training Components
| Component          | Description                                                    | Implementation Details                    |
|-------------------|----------------------------------------------------------------|------------------------------------------|
| `DataProcessor`   | Processes instruction datasets                                | Format conversion and validation         |
| `LossComputer`    | Computes autoregressive loss                                  | Masked token prediction                  |
| `MetricsTracker`  | Monitors training progress                                    | Performance metrics collection           |

## Implementation Details

### Chat Template Implementation
```python
class ChatTemplate:
    def __init__(self, bos_token: str = "<|im_start|>", eos_token: str = "<|im_end|>"):
        self.bos_token = bos_token
        self.eos_token = eos_token
        
    def format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages into a chat template string.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Formatted chat string
        """
        formatted = []
        
        # Handle system message
        if messages[0]['role'] == 'system':
            formatted.append(
                f"{self.bos_token}system\n{messages[0]['content']}{self.eos_token}\n"
            )
            messages = messages[1:]
            
        # Format remaining messages
        for i, message in enumerate(messages):
            role = message['role']
            content = message['content'].strip()
            
            # Validate alternating roles
            if i % 2 == 0 and role != 'user':
                raise ValueError("User messages must alternate with assistant messages")
            if i % 2 == 1 and role != 'assistant':
                raise ValueError("Assistant messages must alternate with user messages")
                
            formatted.append(
                f"{self.bos_token}{role}\n{content}{self.eos_token}\n"
            )
            
        return "".join(formatted)
```

### Training Process
```python
class InstructionTrainer:
    def __init__(self, model, tokenizer, chat_template):
        self.model = model
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        
    def prepare_batch(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """
        Prepare a batch of conversations for training.
        
        Args:
            batch: List of conversation dictionaries
            
        Returns:
            Dictionary of tokenized inputs
        """
        # Format conversations
        formatted = [
            self.chat_template.format_messages(conv)
            for conv in batch
        ]
        
        # Tokenize
        tokenized = self.tokenizer(
            formatted,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Create attention mask for loss computation
        # Only compute loss on assistant responses
        labels = tokenized["input_ids"].clone()
        for i, conv in enumerate(batch):
            # Mask all tokens except assistant responses
            for j, msg in enumerate(conv):
                if msg["role"] != "assistant":
                    labels[i, msg["start_idx"]:msg["end_idx"]] = -100
                    
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels
        }
```

## Best Practices

### Data Quality Guidelines
1. High-quality completions are crucial
2. ~1M prompts sufficient for excellent performance
3. Match prompt distribution to downstream tasks
4. Multiple training stages can recover from noise
5. Mask prompts during loss computation

### Training Configuration
| Parameter     | Type    | Description                                      |
|--------------|---------|--------------------------------------------------|
| `batch_size` | `int`   | Training batch size (default: `32`)              |
| `max_length` | `int`   | Maximum sequence length (default: `2048`)        |
| `learning_rate`| `float`| Learning rate (default: `1e-5`)                 |
| `warmup_steps`| `int`   | Learning rate warmup steps (default: `1000`)     |

### Dataset Requirements
| Requirement   | Description                                                    |
|--------------|----------------------------------------------------------------|
| `size`       | ~1M prompts for optimal performance                            |
| `quality`    | High-quality completions essential                             |
| `diversity`  | Cover target task distribution                                |
| `format`     | Consistent chat template format                               |

## Usage Example
```python
# Initialize components
chat_template = ChatTemplate()
tokenizer = AutoTokenizer.from_pretrained("base_model")
model = AutoModelForCausalLM.from_pretrained("base_model")
trainer = InstructionTrainer(model, tokenizer, chat_template)

# Training loop
for batch in dataloader:
    # Prepare batch
    prepared = trainer.prepare_batch(batch)
    
    # Forward pass
    outputs = model(
        input_ids=prepared["input_ids"],
        attention_mask=prepared["attention_mask"],
        labels=prepared["labels"]
    )
    
    # Backward pass
    loss = outputs.loss
    loss.backward()
    optimizer.step()
```

## Monitoring and Validation

### Key Metrics
| Metric           | Description                                      |
|-----------------|--------------------------------------------------|
| `loss`          | Training loss on assistant responses             |
| `perplexity`    | Model perplexity on validation set              |
| `response_quality`| Quality of generated responses                 |
| `instruction_following`| Adherence to instructions                    |

### Validation Process
1. Regular loss monitoring
2. Response quality assessment
3. Instruction following evaluation
4. Multi-turn conversation testing 