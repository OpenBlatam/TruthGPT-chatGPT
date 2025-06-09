"""
Parallel training optimizations integrated from Paraller.py.
Enhanced PPO actor with distributed training and flash attention support.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass, field
import warnings

@dataclass
class ParallelTrainingConfig:
    """Configuration for parallel training optimizations."""
    micro_batch_size_per_device_for_experience: int = field(default=4)
    micro_batch_size_per_device_for_update: int = field(default=2)
    global_batch_size_per_device: int = field(default=8)
    max_grad_norm: float = field(default=1.0)
    ppo_epochs: int = field(default=1)
    clip_ratio: float = field(default=0.2)
    entropy_coeff: float = field(default=0.01)
    use_kl_loss: bool = field(default=False)
    kl_loss_coef: float = field(default=0.1)
    kl_loss_type: str = field(default="kl")
    padding_free: bool = field(default=False)
    ulysses_sequence_parallel_size: int = field(default=1)
    logging_steps: int = field(default=10)

class MockDataProto:
    """Mock data protocol for compatibility."""
    def __init__(self, batch_data, non_tensor_batch=None, meta_info=None):
        self.batch = batch_data
        self.non_tensor_batch = non_tensor_batch or {}
        self.meta_info = meta_info or {}
    
    def select(self, keys, non_tensor_keys=None):
        selected_batch = {k: self.batch[k] for k in keys if k in self.batch}
        selected_non_tensor = {}
        if non_tensor_keys:
            selected_non_tensor = {k: self.non_tensor_batch[k] for k in non_tensor_keys if k in self.non_tensor_batch}
        return MockDataProto(selected_batch, selected_non_tensor, self.meta_info)
    
    def split(self, batch_size):
        batches = []
        total_size = len(next(iter(self.batch.values())))
        for i in range(0, total_size, batch_size):
            split_batch = {}
            for k, v in self.batch.items():
                if isinstance(v, torch.Tensor):
                    split_batch[k] = v[i:i+batch_size]
                else:
                    split_batch[k] = v[i:i+batch_size]
            batches.append(MockDataProto(split_batch, self.non_tensor_batch, self.meta_info))
        return batches

def masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute masked mean of tensor."""
    return (tensor * mask).sum() / mask.sum().clamp(min=1)

def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy from logits."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)

def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute log probabilities from logits and labels."""
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

class MockCoreAlgos:
    """Mock core algorithms for compatibility."""
    
    @staticmethod
    def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
        """Compute PPO policy loss."""
        ratio = torch.exp(log_prob - old_log_prob)
        clipped_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
        
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * clipped_ratio
        pg_loss = torch.max(pg_loss1, pg_loss2)
        pg_loss = masked_mean(pg_loss, eos_mask)
        
        pg_clipfrac = ((ratio - clipped_ratio).abs() > 1e-6).float()
        pg_clipfrac = masked_mean(pg_clipfrac, eos_mask)
        
        ppo_kl = masked_mean(old_log_prob - log_prob, eos_mask)
        
        return pg_loss, pg_clipfrac, ppo_kl
    
    @staticmethod
    def kl_penalty(logprob, ref_logprob, kl_penalty="kl"):
        """Compute KL penalty."""
        if kl_penalty == "kl":
            return ref_logprob - logprob
        elif kl_penalty == "abs":
            return (ref_logprob - logprob).abs()
        else:
            return ref_logprob - logprob

core_algos = MockCoreAlgos()

class EnhancedPPOActor:
    """Enhanced PPO Actor with parallel training optimizations."""
    
    def __init__(
        self,
        config: ParallelTrainingConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.config = config
        self.rank = int(os.getenv("RANK", "0"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        
        try:
            self.compute_entropy_from_logits = torch.compile(entropy_from_logits, dynamic=True)
        except:
            self.compute_entropy_from_logits = entropy_from_logits
            warnings.warn("torch.compile not available, using standard entropy computation")

    def _forward_micro_batch(
        self, micro_batch: Dict[str, torch.Tensor], temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a micro batch with optimizations."""
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch.get("position_ids")
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        
        if position_ids is not None and position_ids.dim() == 3:
            position_ids = position_ids.transpose(0, 1)

        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                )

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            if self.config.padding_free:
                return self._forward_padding_free(
                    input_ids, attention_mask, position_ids, responses, 
                    multi_modal_inputs, temperature, response_length
                )
            else:
                return self._forward_standard(
                    input_ids, attention_mask, position_ids, responses,
                    multi_modal_inputs, temperature, response_length
                )

    def _forward_padding_free(self, input_ids, attention_mask, position_ids, responses,
                            multi_modal_inputs, temperature, response_length):
        """Forward pass with padding-free optimization."""
        try:
            from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
        except ImportError:
            warnings.warn("flash_attn not available, falling back to standard attention")
            return self._forward_standard(
                input_ids, attention_mask, position_ids, responses,
                multi_modal_inputs, temperature, response_length
            )
        
        batch_size, seqlen = input_ids.shape
        
        input_ids_rmpad, indices, *_ = unpad_input(
            input_ids.unsqueeze(-1), attention_mask
        )
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

        if position_ids is not None:
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)
        else:
            position_ids_rmpad = None

        input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

        if self.config.ulysses_sequence_parallel_size > 1:
            input_ids_rmpad, position_ids_rmpad, pad_size = self._ulysses_pad_and_slice(
                input_ids_rmpad, position_ids_rmpad
            )
            input_ids_rmpad_rolled, _, _ = self._ulysses_pad_and_slice(
                input_ids_rmpad_rolled, None
            )

        input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)

        output = self.actor_module(
            input_ids=input_ids_rmpad,
            attention_mask=None,
            position_ids=position_ids_rmpad,
            **multi_modal_inputs,
            use_cache=False,
        )
        logits_rmpad = output.logits.squeeze(0)
        logits_rmpad.div_(temperature)

        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)
        log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

        if self.config.ulysses_sequence_parallel_size > 1:
            log_probs = self._gather_and_unpad(log_probs, pad_size)
            entropy_rmpad = self._gather_and_unpad(entropy_rmpad, pad_size)

        full_entropy = pad_input(
            hidden_states=entropy_rmpad.unsqueeze(-1), 
            indices=indices, 
            batch=batch_size, 
            seqlen=seqlen
        )
        full_log_probs = pad_input(
            hidden_states=log_probs.unsqueeze(-1), 
            indices=indices, 
            batch=batch_size, 
            seqlen=seqlen
        )

        entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]
        log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]

        return entropy, log_probs

    def _forward_standard(self, input_ids, attention_mask, position_ids, responses,
                         multi_modal_inputs, temperature, response_length):
        """Standard forward pass without padding-free optimization."""
        output = self.actor_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **multi_modal_inputs,
            use_cache=False,
        )
        logits = output.logits
        logits.div_(temperature)
        logits = logits[:, -response_length - 1 : -1, :]
        log_probs = logprobs_from_logits(logits, responses)
        entropy = entropy_from_logits(logits)

        return entropy, log_probs

    def _ulysses_pad_and_slice(self, input_tensor, position_tensor):
        """Mock implementation of Ulysses sequence parallelism."""
        pad_size = 0
        return input_tensor, position_tensor, pad_size

    def _gather_and_unpad(self, tensor, pad_size):
        """Mock implementation of gather and unpad for sequence parallelism."""
        return tensor

    def _optimizer_step(self) -> torch.Tensor:
        """Optimizer step with gradient clipping."""
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(
                self.actor_module.parameters(), 
                max_norm=self.config.max_grad_norm
            )

        self.actor_optimizer.step()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data) -> torch.Tensor:
        """Compute log probabilities for given data."""
        self.actor_module.eval()

        temperature = data.meta_info.get("temperature", 1.0)
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = []
        
        if "multi_modal_inputs" in data.non_tensor_batch:
            non_tensor_select_keys = ["multi_modal_inputs"]

        micro_batches = data.select(select_keys, non_tensor_select_keys).split(
            self.config.micro_batch_size_per_device_for_experience
        )
        
        log_probs_lst = []
        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            _, log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        return log_probs

    def update_policy(self, data) -> Dict[str, Any]:
        """Update policy using PPO with enhanced optimizations."""
        self.actor_module.train()

        temperature = data.meta_info.get("temperature", 1.0)
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        non_tensor_select_keys = []
        if "multi_modal_inputs" in data.non_tensor_batch:
            non_tensor_select_keys = ["multi_modal_inputs"]

        mini_batches = data.select(select_keys, non_tensor_select_keys).split(
            self.config.global_batch_size_per_device
        )

        metrics = defaultdict(list)
        n = len(mini_batches)
        
        for _ in range(self.config.ppo_epochs):
            for i, mini_batch in enumerate(mini_batches):
                gradient_accumulation = (
                    self.config.global_batch_size_per_device // 
                    self.config.micro_batch_size_per_device_for_update
                )
                micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

                self.actor_optimizer.zero_grad()
                
                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    responses = model_inputs["responses"]
                    response_length = responses.size(1)
                    attention_mask = model_inputs["attention_mask"]
                    response_mask = attention_mask[:, -response_length:]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy, log_prob = self._forward_micro_batch(model_inputs, temperature=temperature)

                    pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        eos_mask=response_mask,
                        cliprange=self.config.clip_ratio,
                    )
                    
                    entropy_loss = masked_mean(entropy, response_mask)

                    policy_loss = pg_loss - entropy_loss * self.config.entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kld = core_algos.kl_penalty(
                            logprob=log_prob,
                            ref_logprob=ref_log_prob,
                            kl_penalty=self.config.kl_loss_type,
                        )
                        kl_loss = masked_mean(kld, response_mask)
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"].append(kl_loss.detach().item())
                        metrics["actor/kl_coef"].append(self.config.kl_loss_coef)

                    loss = policy_loss / gradient_accumulation
                    loss.backward()

                    batch_metrics = {
                        "actor/entropy_loss": entropy_loss.detach().item(),
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                    }
                    for k, v in batch_metrics.items():
                        metrics[k].append(v)

                grad_norm = self._optimizer_step()
                metrics["actor/grad_norm"].append(grad_norm.detach().item())

        self.actor_optimizer.zero_grad()
        return {k: sum(v) / len(v) for k, v in metrics.items()}

def create_parallel_actor(model: nn.Module, config: Optional[ParallelTrainingConfig] = None, 
                         optimizer: Optional[torch.optim.Optimizer] = None):
    """Factory function to create enhanced parallel PPO actor."""
    if config is None:
        config = ParallelTrainingConfig()
    return EnhancedPPOActor(config, model, optimizer)

def setup_distributed_training(backend: str = "nccl", init_method: str = "env://"):
    """Setup distributed training environment."""
    if torch.cuda.is_available() and torch.distributed.is_available():
        try:
            init_process_group(backend=backend, init_method=init_method)
            return True
        except Exception as e:
            warnings.warn(f"Failed to initialize distributed training: {e}")
            return False
    return False

def cleanup_distributed_training():
    """Cleanup distributed training environment."""
    if torch.distributed.is_initialized():
        destroy_process_group()

def wrap_model_for_distributed(model: nn.Module, use_fsdp: bool = False):
    """Wrap model for distributed training."""
    if not torch.distributed.is_initialized():
        return model
    
    if use_fsdp and torch.cuda.is_available():
        try:
            return FSDP(model)
        except Exception as e:
            warnings.warn(f"Failed to wrap with FSDP: {e}, falling back to DDP")
    
    return DDP(model)
