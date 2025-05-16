# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import dataclass, field
from typing import List, Any, Dict

import datasets
import torch
import transformers
from datasets import load_dataset, DatasetDict
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig
from open_r1.rewards import (
    accuracy_reward,
    code_reward,
    format_reward,
    get_code_format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    len_reward,
    reasoning_steps_reward,
    tag_count_reward,
)
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from rich.logging import RichHandler
from rich.traceback import install as rich_traceback_install
import logging

def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
    rich_traceback_install()
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[RichHandler()]
    )
    return logging.getLogger("rich")

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """Script arguments for the GRPO training script."""
    reward_funcs: List[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
        },
    )
    cosine_min_value_wrong: float = field(default=0.0)
    cosine_max_value_wrong: float = field(default=-0.5)
    cosine_min_value_correct: float = field(default=0.5)
    cosine_max_value_correct: float = field(default=1.0)
    cosine_max_len: int = field(default=1000)
    repetition_n_grams: int = field(default=3)
    repetition_max_penalty: float = field(default=-1.0)
    code_language: str = field(
        default="python",
        metadata={
            "help": "Language for code format reward. Based on E2B supported languages https://e2b.dev/docs/code-interpreting/supported-languages",
            "choices": ["python", "javascript", "r", "java", "bash"],
        },
    )

REWARD_FUNCS_REGISTRY = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "reasoning_steps": reasoning_steps_reward,
    "cosine": lambda args: get_cosine_scaled_reward(
        min_value_wrong=args.cosine_min_value_wrong,
        max_value_wrong=args.cosine_max_value_wrong,
        min_value_correct=args.cosine_min_value_correct,
        max_value_correct=args.cosine_max_value_correct,
        max_len=args.cosine_max_len,
    ),
    "repetition_penalty": lambda args: get_repetition_penalty_reward(
        ngram_size=args.repetition_n_grams,
        max_penalty=args.repetition_max_penalty,
    ),
    "length": len_reward,
    "code": code_reward,
    "code_format": lambda args: get_code_format_reward(language=args.code_language),
    "tag_count": tag_count_reward,
}

def get_reward_funcs(args: GRPOScriptArguments) -> List[Any]:
    funcs = []
    for func_name in args.reward_funcs:
        if func_name not in REWARD_FUNCS_REGISTRY:
            raise ValueError(f"Unknown reward function: {func_name}")
        func = REWARD_FUNCS_REGISTRY[func_name]
        if callable(func) and hasattr(func, '__call__') and func.__code__.co_argcount:
            try:
                funcs.append(func(args))
            except Exception:
                funcs.append(func)
        else:
            funcs.append(func)
    return funcs

def make_conversation(example: Dict, system_prompt: str = None) -> Dict:
    prompt = []
    if system_prompt is not None:
        prompt.append({"role": "system", "content": system_prompt})
    prompt.append({"role": "user", "content": example["problem"]})
    return {"prompt": prompt}

def load_and_prepare_dataset(script_args: GRPOScriptArguments, training_args: Any, logger: logging.Logger) -> DatasetDict:
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    logger.info("Formatting dataset into conversations...")
    dataset = dataset.map(
        lambda ex: make_conversation(ex, training_args.system_prompt),
        num_proc=4,
        desc="Formatting",
        batch_size=100,
        batched=True,
    )
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")
    return dataset

def build_trainer(model_args: Any, training_args: Any, script_args: GRPOScriptArguments, reward_funcs: List[Any], dataset: DatasetDict, tokenizer: Any, logger: logging.Logger) -> GRPOTrainer:
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs
    return GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

def train_and_evaluate(trainer: GRPOTrainer, dataset: DatasetDict, script_args: GRPOScriptArguments, training_args: Any, last_checkpoint: Any, logger: logging.Logger) -> None:
    logger.info("*** Train ***")
    checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

def save_and_push(trainer: GRPOTrainer, training_args: Any, script_args: GRPOScriptArguments, logger: logging.Logger) -> None:
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

def main(script_args: GRPOScriptArguments, training_args: Any, model_args: Any) -> None:
    logger = setup_logging(training_args.get_process_log_level())
    set_seed(training_args.seed)
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")
    last_checkpoint = get_last_checkpoint(training_args.output_dir) if os.path.isdir(training_args.output_dir) else None
    if last_checkpoint and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")
    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)
    dataset = load_and_prepare_dataset(script_args, training_args, logger)
    tokenizer = get_tokenizer(model_args, training_args)
    reward_funcs = get_reward_funcs(script_args)
    trainer = build_trainer(model_args, training_args, script_args, reward_funcs, dataset, tokenizer, logger)
    train_and_evaluate(trainer, dataset, script_args, training_args, last_checkpoint, logger)
    save_and_push(trainer, training_args, script_args, logger)

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args) 