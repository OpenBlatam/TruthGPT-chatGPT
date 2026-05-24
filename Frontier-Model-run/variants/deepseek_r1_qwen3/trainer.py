"""
DeepSeek-R1-Qwen3 Advanced Reasoning Trainer

Specialized trainer for the DeepSeek-R1-Qwen3 model with advanced reasoning capabilities,
chain-of-thought training, and multi-objective optimization.

Key Features:
- Reasoning-aware training loops
- Chain-of-thought optimization
- Step-by-step verification training
- Confidence calibration
- Multi-objective loss functions
"""

import os
import json
import math
import time
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    get_scheduler,
)
from transformers.trainer_utils import EvalPrediction
import numpy as np
from tqdm.auto import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

logger = logging.getLogger(__name__)


@dataclass
class ReasoningTrainingArguments(TrainingArguments):
    """Extended training arguments for reasoning training."""
    
    # Reasoning-specific parameters
    reasoning_loss_weight: float = field(default=0.3, metadata={"help": "Weight for reasoning loss"})
    verification_loss_weight: float = field(default=0.2, metadata={"help": "Weight for verification loss"})
    confidence_loss_weight: float = field(default=0.1, metadata={"help": "Weight for confidence loss"})
    step_consistency_weight: float = field(default=0.15, metadata={"help": "Weight for step consistency loss"})
    
    # Chain-of-thought training
    cot_training: bool = field(default=True, metadata={"help": "Enable chain-of-thought training"})
    cot_data_ratio: float = field(default=0.4, metadata={"help": "Ratio of CoT data in training"})
    max_cot_length: int = field(default=2048, metadata={"help": "Maximum CoT sequence length"})
    cot_temperature: float = field(default=0.7, metadata={"help": "Temperature for CoT generation"})
    
    # Reasoning curriculum
    curriculum_learning: bool = field(default=True, metadata={"help": "Enable curriculum learning"})
    start_reasoning_epoch: int = field(default=1, metadata={"help": "Epoch to start reasoning training"})
    reasoning_difficulty_schedule: str = field(default="linear", metadata={"help": "Difficulty schedule"})
    
    # Evaluation parameters
    eval_reasoning_steps: int = field(default=100, metadata={"help": "Steps between reasoning evaluations"})
    save_reasoning_traces: bool = field(default=True, metadata={"help": "Save reasoning traces during evaluation"})


class ReasoningDataset(Dataset):
    """Dataset for reasoning training with chain-of-thought support."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 4096,
        max_reasoning_length: int = 2048,
        reasoning_format: str = "step_by_step",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_reasoning_length = max_reasoning_length
        self.reasoning_format = reasoning_format
        
        # Load data
        self.examples = self._load_data(data_path)
        logger.info(f"Loaded {len(self.examples)} reasoning examples from {data_path}")
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load reasoning data from JSONL file."""
        examples = []
        
        if not os.path.exists(data_path):
            logger.warning(f"Data path {data_path} does not exist. Creating dummy data.")
            return self._create_dummy_data()
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    example = json.loads(line.strip())
                    if self._validate_example(example):
                        examples.append(example)
                except json.JSONDecodeError:
                    continue
        
        return examples
    
    def _create_dummy_data(self) -> List[Dict]:
        """Create dummy reasoning data for testing."""
        dummy_examples = [
            {
                "problem": "What is 15 * 23?",
                "reasoning_steps": [
                    "I need to multiply 15 by 23.",
                    "Let me break this down: 15 * 23 = 15 * (20 + 3) = 15 * 20 + 15 * 3",
                    "15 * 20 = 300",
                    "15 * 3 = 45", 
                    "So 15 * 23 = 300 + 45 = 345"
                ],
                "answer": "345",
                "verification": "correct",
                "confidence": 0.95,
                "difficulty": "easy"
            },
            {
                "problem": "If a train travels 120 km in 2 hours, what is its average speed?",
                "reasoning_steps": [
                    "I need to find the average speed of the train.",
                    "Average speed = Total distance / Total time",
                    "Total distance = 120 km",
                    "Total time = 2 hours",
                    "Average speed = 120 km / 2 hours = 60 km/h"
                ],
                "answer": "60 km/h",
                "verification": "correct",
                "confidence": 0.9,
                "difficulty": "medium"
            },
            {
                "problem": "Solve for x: 2x + 5 = 13",
                "reasoning_steps": [
                    "I need to solve the equation 2x + 5 = 13 for x.",
                    "First, I'll subtract 5 from both sides: 2x + 5 - 5 = 13 - 5",
                    "This gives me: 2x = 8",
                    "Now I'll divide both sides by 2: 2x / 2 = 8 / 2",
                    "Therefore: x = 4"
                ],
                "answer": "x = 4",
                "verification": "correct",
                "confidence": 0.98,
                "difficulty": "medium"
            }
        ]
        
        # Replicate to create more examples
        examples = []
        for i in range(100):
            for dummy in dummy_examples:
                example = dummy.copy()
                example["id"] = f"dummy_{i}_{dummy['problem'][:10]}"
                examples.append(example)
        
        return examples
    
    def _validate_example(self, example: Dict) -> bool:
        """Validate that an example has required fields."""
        required_fields = ["problem", "reasoning_steps", "answer"]
        return all(field in example for field in required_fields)
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Format the input
        problem = example["problem"]
        reasoning_steps = example.get("reasoning_steps", [])
        answer = example["answer"]
        
        # Create reasoning chain
        if self.reasoning_format == "step_by_step":
            reasoning_text = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(reasoning_steps)])
            full_text = f"Problem: {problem}\n\nReasoning:\n{reasoning_text}\n\nAnswer: {answer}"
        else:
            reasoning_text = " ".join(reasoning_steps)
            full_text = f"Problem: {problem}\nThinking: {reasoning_text}\nAnswer: {answer}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create labels (same as input_ids for causal LM)
        labels = encoding["input_ids"].clone()
        
        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Additional metadata
        metadata = {
            "verification": example.get("verification", "unknown"),
            "confidence": example.get("confidence", 0.5),
            "difficulty": example.get("difficulty", "medium"),
            "num_reasoning_steps": len(reasoning_steps),
        }
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
            **metadata
        }


class ReasoningLossComputer:
    """Computes multi-objective losses for reasoning training."""
    
    def __init__(self, config):
        self.config = config
        self.reasoning_weight = config.reasoning_loss_weight
        self.verification_weight = config.verification_loss_weight
        self.confidence_weight = config.confidence_loss_weight
        self.consistency_weight = config.step_consistency_weight
    
    def compute_reasoning_loss(
        self,
        model_outputs,
        labels: torch.Tensor,
        metadata: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-objective reasoning loss."""
        losses = {}
        
        # Standard language modeling loss
        if hasattr(model_outputs, 'loss') and model_outputs.loss is not None:
            losses["lm_loss"] = model_outputs.loss
        else:
            # Compute manually if not provided
            logits = model_outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            losses["lm_loss"] = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Reasoning-specific losses
        if hasattr(model_outputs, 'reasoning_outputs') and model_outputs.reasoning_outputs:
            reasoning_outputs = model_outputs.reasoning_outputs
            
            # Verification loss
            if reasoning_outputs and len(reasoning_outputs) > 0:
                verification_losses = []
                confidence_losses = []
                
                for reasoning_output in reasoning_outputs:
                    if "verification_probs" in reasoning_output:
                        # Create verification targets based on metadata
                        verification_targets = self._create_verification_targets(metadata)
                        if verification_targets is not None:
                            verification_loss = F.cross_entropy(
                                reasoning_output["verification_probs"].view(-1, 2),
                                verification_targets.view(-1)
                            )
                            verification_losses.append(verification_loss)
                    
                    if "confidence" in reasoning_output:
                        # Confidence calibration loss
                        target_confidence = metadata.get("confidence", torch.tensor(0.5))
                        if isinstance(target_confidence, (int, float)):
                            target_confidence = torch.tensor(target_confidence, device=reasoning_output["confidence"].device)
                        confidence_loss = F.mse_loss(
                            reasoning_output["confidence"].mean(),
                            target_confidence
                        )
                        confidence_losses.append(confidence_loss)
                
                if verification_losses:
                    losses["verification_loss"] = torch.stack(verification_losses).mean()
                
                if confidence_losses:
                    losses["confidence_loss"] = torch.stack(confidence_losses).mean()
        
        # Step consistency loss (encourage consistent reasoning across steps)
        if hasattr(model_outputs, 'reasoning_outputs') and len(model_outputs.reasoning_outputs) > 1:
            consistency_loss = self._compute_step_consistency_loss(model_outputs.reasoning_outputs)
            losses["consistency_loss"] = consistency_loss
        
        # Combine losses
        total_loss = losses["lm_loss"]
        
        if "verification_loss" in losses:
            total_loss += self.verification_weight * losses["verification_loss"]
        
        if "confidence_loss" in losses:
            total_loss += self.confidence_weight * losses["confidence_loss"]
        
        if "consistency_loss" in losses:
            total_loss += self.consistency_weight * losses["consistency_loss"]
        
        losses["total_loss"] = total_loss
        
        return losses
    
    def _create_verification_targets(self, metadata: Dict) -> Optional[torch.Tensor]:
        """Create verification targets from metadata."""
        if "verification" not in metadata:
            return None
        
        verification = metadata["verification"]
        if isinstance(verification, str):
            # Convert string to tensor
            if verification == "correct":
                return torch.tensor(1, dtype=torch.long)
            elif verification == "incorrect":
                return torch.tensor(0, dtype=torch.long)
        
        return None
    
    def _compute_step_consistency_loss(self, reasoning_outputs: List[Dict]) -> torch.Tensor:
        """Compute consistency loss across reasoning steps."""
        if len(reasoning_outputs) < 2:
            return torch.tensor(0.0)
        
        consistency_losses = []
        
        for i in range(len(reasoning_outputs) - 1):
            current_features = reasoning_outputs[i].get("step_features")
            next_features = reasoning_outputs[i + 1].get("step_features")
            
            if current_features is not None and next_features is not None:
                # Encourage smooth transitions between reasoning steps
                consistency_loss = F.mse_loss(current_features, next_features)
                consistency_losses.append(consistency_loss)
        
        if consistency_losses:
            return torch.stack(consistency_losses).mean()
        
        return torch.tensor(0.0)


class ReasoningTrainer(Trainer):
    """Specialized trainer for reasoning models."""
    
    def __init__(
        self,
        model,
        args: ReasoningTrainingArguments,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        data_collator=None,
        compute_metrics=None,
        **kwargs
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            **kwargs
        )
        
        self.reasoning_args = args
        self.loss_computer = ReasoningLossComputer(args)
        self.reasoning_metrics = defaultdict(list)
        
        # Initialize curriculum learning
        if args.curriculum_learning:
            self.curriculum_scheduler = self._init_curriculum_scheduler()
    
    def _init_curriculum_scheduler(self):
        """Initialize curriculum learning scheduler."""
        total_steps = self.args.max_steps if self.args.max_steps > 0 else len(self.train_dataloader) * self.args.num_train_epochs
        
        if self.reasoning_args.reasoning_difficulty_schedule == "linear":
            return lambda step: min(1.0, step / (total_steps * 0.8))
        elif self.reasoning_args.reasoning_difficulty_schedule == "exponential":
            return lambda step: 1.0 - math.exp(-step / (total_steps * 0.3))
        else:  # step
            return lambda step: 1.0 if step > total_steps * 0.5 else 0.5
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute multi-objective reasoning loss."""
        # Extract metadata
        metadata = {}
        for key in ["verification", "confidence", "difficulty", "num_reasoning_steps"]:
            if key in inputs:
                metadata[key] = inputs.pop(key)
        
        # Forward pass with reasoning mode
        if self.state.epoch >= self.reasoning_args.start_reasoning_epoch:
            inputs["reasoning_mode"] = True
        
        outputs = model(**inputs)
        
        # Compute losses
        losses = self.loss_computer.compute_reasoning_loss(outputs, inputs["labels"], metadata)
        
        # Log individual losses
        if self.state.is_world_process_zero:
            for loss_name, loss_value in losses.items():
                if loss_name != "total_loss":
                    self.log({f"train/{loss_name}": loss_value.item()})
        
        loss = losses["total_loss"]
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        """Enhanced evaluation loop with reasoning metrics."""
        # Standard evaluation
        output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )
        
        # Additional reasoning evaluation
        if not prediction_loss_only:
            reasoning_metrics = self._evaluate_reasoning(dataloader)
            output.metrics.update({f"{metric_key_prefix}_{k}": v for k, v in reasoning_metrics.items()})
        
        return output
    
    def _evaluate_reasoning(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate reasoning capabilities."""
        model = self.model
        model.eval()
        
        reasoning_metrics = {
            "reasoning_accuracy": 0.0,
            "avg_confidence": 0.0,
            "avg_reasoning_steps": 0.0,
            "verification_accuracy": 0.0,
        }
        
        total_examples = 0
        correct_reasoning = 0
        total_confidence = 0.0
        total_steps = 0
        correct_verifications = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Reasoning evaluation"):
                # Move to device
                batch = {k: v.to(self.args.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Extract metadata
                metadata = {}
                for key in ["verification", "confidence", "difficulty", "num_reasoning_steps"]:
                    if key in batch:
                        metadata[key] = batch.pop(key)
                
                # Forward pass with reasoning
                batch["reasoning_mode"] = True
                outputs = model(**batch)
                
                batch_size = batch["input_ids"].size(0)
                total_examples += batch_size
                
                # Analyze reasoning outputs
                if hasattr(outputs, 'reasoning_outputs') and outputs.reasoning_outputs:
                    for i, reasoning_output in enumerate(outputs.reasoning_outputs):
                        if "confidence" in reasoning_output:
                            confidence = reasoning_output["confidence"].mean().item()
                            total_confidence += confidence
                            
                            # Simple reasoning accuracy based on confidence threshold
                            if confidence > 0.7:
                                correct_reasoning += 1
                        
                        if "verification_probs" in reasoning_output:
                            verification_pred = reasoning_output["verification_probs"].argmax(dim=-1)
                            if "verification" in metadata:
                                verification_target = metadata["verification"]
                                if isinstance(verification_target, str):
                                    target_val = 1 if verification_target == "correct" else 0
                                    if verification_pred.item() == target_val:
                                        correct_verifications += 1
                
                if "num_reasoning_steps" in metadata:
                    total_steps += metadata["num_reasoning_steps"]
        
        if total_examples > 0:
            reasoning_metrics["reasoning_accuracy"] = correct_reasoning / total_examples
            reasoning_metrics["avg_confidence"] = total_confidence / total_examples
            reasoning_metrics["avg_reasoning_steps"] = total_steps / total_examples
            reasoning_metrics["verification_accuracy"] = correct_verifications / total_examples
        
        return reasoning_metrics
    
    def log(self, logs: Dict[str, float]) -> None:
        """Enhanced logging with reasoning metrics."""
        # Add reasoning-specific metrics
        if hasattr(self, 'reasoning_metrics'):
            for key, values in self.reasoning_metrics.items():
                if values:
                    logs[f"reasoning/{key}"] = np.mean(values[-10:])  # Average of last 10 values
        
        super().log(logs)
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Save model with reasoning configuration."""
        super().save_model(output_dir, _internal_call)
        
        # Save reasoning-specific configuration
        if output_dir is not None and self.state.is_world_process_zero:
            reasoning_config = {
                "reasoning_args": self.reasoning_args.to_dict(),
                "reasoning_metrics": dict(self.reasoning_metrics),
                "curriculum_progress": getattr(self, 'curriculum_progress', 0.0),
            }
            
            config_path = os.path.join(output_dir, "reasoning_config.json")
            with open(config_path, 'w') as f:
                json.dump(reasoning_config, f, indent=2)


def create_reasoning_data_collator(tokenizer, max_length: int = 4096):
    """Create data collator for reasoning training."""
    
    def collate_fn(examples):
        # Standard collation
        batch = {}
        
        # Collect all keys
        all_keys = set()
        for example in examples:
            all_keys.update(example.keys())
        
        for key in all_keys:
            if key in ["input_ids", "attention_mask", "labels"]:
                # Tensor fields
                values = [example[key] for example in examples if key in example]
                if values:
                    batch[key] = torch.stack(values)
            else:
                # Metadata fields
                values = [example.get(key, None) for example in examples]
                if all(v is not None for v in values):
                    if isinstance(values[0], (int, float)):
                        batch[key] = torch.tensor(values)
                    else:
                        batch[key] = values
        
        return batch
    
    return collate_fn


def compute_reasoning_metrics(eval_preds: EvalPrediction) -> Dict[str, float]:
    """Compute reasoning-specific metrics."""
    predictions, labels = eval_preds
    
    # Basic perplexity
    predictions = predictions.reshape(-1, predictions.shape[-1])
    labels = labels.reshape(-1)
    
    # Filter out ignored tokens
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]
    
    # Compute perplexity
    loss = F.cross_entropy(torch.from_numpy(predictions), torch.from_numpy(labels))
    perplexity = torch.exp(loss).item()
    
    return {
        "perplexity": perplexity,
        "eval_loss": loss.item(),
    }


class ReasoningEvaluator:
    """Evaluator for reasoning capabilities on specific benchmarks."""
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
    def evaluate_on_benchmark(
        self,
        benchmark_name: str,
        benchmark_data: List[Dict],
        max_tokens: int = 1024,
        temperature: float = 0.6,
    ) -> Dict[str, float]:
        """Evaluate model on a specific reasoning benchmark."""
        self.model.eval()
        
        results = {
            "accuracy": 0.0,
            "avg_confidence": 0.0,
            "avg_reasoning_steps": 0.0,
            "pass_at_1": 0.0,
        }
        
        correct = 0
        total_confidence = 0.0
        total_steps = 0
        
        with torch.no_grad():
            for example in tqdm(benchmark_data, desc=f"Evaluating {benchmark_name}"):
                # Generate response with reasoning
                response = self._generate_with_reasoning(
                    example["problem"],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                
                # Check correctness
                if self._check_answer_correctness(response["generated_text"], example["answer"]):
                    correct += 1
                
                total_confidence += response.get("avg_confidence", 0.5)
                total_steps += response.get("num_reasoning_steps", 0)
        
        total_examples = len(benchmark_data)
        if total_examples > 0:
            results["accuracy"] = correct / total_examples
            results["pass_at_1"] = correct / total_examples
            results["avg_confidence"] = total_confidence / total_examples
            results["avg_reasoning_steps"] = total_steps / total_examples
        
        return results
    
    def _generate_with_reasoning(
        self,
        problem: str,
        max_tokens: int = 1024,
        temperature: float = 0.6,
    ) -> Dict[str, Any]:
        """Generate response with reasoning analysis."""
        # Tokenize input
        inputs = self.tokenizer(
            f"Problem: {problem}\n\nReasoning:",
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)
        
        # Generate with reasoning
        outputs = self.model.generate_with_reasoning(
            input_ids=inputs["input_ids"],
            max_length=inputs["input_ids"].shape[1] + max_tokens,
            temperature=temperature,
            confidence_threshold=0.7,
        )
        
        # Decode response
        generated_text = self.tokenizer.decode(
            outputs["generated_ids"][0],
            skip_special_tokens=True,
        )
        
        return {
            "generated_text": generated_text,
            "reasoning_steps": outputs.get("reasoning_steps", []),
            "num_reasoning_steps": outputs.get("num_reasoning_steps", 0),
            "avg_confidence": np.mean([step.get("confidence", 0.5) for step in outputs.get("reasoning_steps", [])]),
        }
    
    def _check_answer_correctness(self, generated_text: str, correct_answer: str) -> bool:
        """Check if generated answer is correct."""
        # Simple string matching (can be enhanced with more sophisticated methods)
        generated_lower = generated_text.lower().strip()
        correct_lower = correct_answer.lower().strip()
        
        # Check if correct answer is contained in generated text
        return correct_lower in generated_lower


def setup_reasoning_training(
    model,
    tokenizer,
    config_path: str,
    train_data_path: str,
    eval_data_path: str,
    output_dir: str,
) -> Tuple[ReasoningTrainer, ReasoningTrainingArguments]:
    """Setup complete reasoning training pipeline."""
    
    # Load configuration
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    training_config = config["training_config"]
    data_config = config["data_config"]
    
    # Create training arguments
    args = ReasoningTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        warmup_ratio=training_config["warmup_ratio"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        
        # Reasoning-specific parameters
        reasoning_loss_weight=training_config["reasoning_training"]["reasoning_loss_weight"],
        verification_loss_weight=training_config["reasoning_training"]["verification_loss_weight"],
        confidence_loss_weight=training_config["reasoning_training"]["confidence_loss_weight"],
        step_consistency_weight=training_config["reasoning_training"]["step_consistency_weight"],
        
        cot_training=training_config["reasoning_training"]["cot_training"],
        cot_data_ratio=training_config["reasoning_training"]["cot_data_ratio"],
        max_cot_length=training_config["reasoning_training"]["max_cot_length"],
        cot_temperature=training_config["reasoning_training"]["cot_temperature"],
        
        curriculum_learning=training_config["reasoning_training"]["curriculum_learning"],
        start_reasoning_epoch=training_config["reasoning_training"]["start_reasoning_epoch"],
        reasoning_difficulty_schedule=training_config["reasoning_training"]["reasoning_difficulty_schedule"],
        
        # Standard training parameters
        logging_steps=training_config["logging_steps"],
        eval_steps=training_config["eval_steps"],
        save_steps=training_config["save_steps"],
        save_total_limit=training_config["save_total_limit"],
        evaluation_strategy=training_config["evaluation_strategy"],
        save_strategy=training_config["save_strategy"],
        load_best_model_at_end=training_config["load_best_model_at_end"],
        metric_for_best_model=training_config["metric_for_best_model"],
        greater_is_better=training_config["greater_is_better"],
        
        # Mixed precision
        bf16=training_config["bf16"],
        gradient_checkpointing=training_config["gradient_checkpointing"],
        
        # Logging
        report_to=training_config.get("report_to", "none"),
        run_name=training_config.get("run_name", "reasoning_training"),
    )
    
    # Create datasets
    train_dataset = ReasoningDataset(
        train_data_path,
        tokenizer,
        max_length=data_config["max_seq_length"],
        max_reasoning_length=data_config["max_reasoning_length"],
    )
    
    eval_dataset = ReasoningDataset(
        eval_data_path,
        tokenizer,
        max_length=data_config["max_seq_length"],
        max_reasoning_length=data_config["max_reasoning_length"],
    )
    
    # Create data collator
    data_collator = create_reasoning_data_collator(tokenizer, data_config["max_seq_length"])
    
    # Create trainer
    trainer = ReasoningTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_reasoning_metrics,
    )
    
    return trainer, args