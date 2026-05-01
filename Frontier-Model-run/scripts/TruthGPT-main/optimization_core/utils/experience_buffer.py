"""
Experience buffer and replay system for enhanced training.
Integrated from buffer.py optimization file.
"""

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Self, List, Any, Dict
import warnings

def zero_pad_sequences(
    sequences: list[torch.Tensor], side: str = "left"
) -> torch.Tensor:
    """Pad sequences to the same length."""
    assert side in ("left", "right")
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding))
    return torch.stack(padded_sequences, dim=0)

class Experience(BaseModel):
    """Experience data structure for replay buffer (Pydantic-First)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    log_probs_ref: torch.Tensor
    action_mask: torch.Tensor
    returns: Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    kl: Optional[torch.Tensor] = None
    rewards: Optional[torch.Tensor] = None
    values: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> Self:
        """Move experience to specified device."""
        updates = {}
        for name, value in self.model_dump().items():
            if isinstance(value, torch.Tensor):
                updates[name] = value.to(device=device)
            else:
                updates[name] = value
        return self.model_copy(update=updates)

    def detach(self) -> Self:
        """Detach all tensors from computation graph."""
        updates = {}
        for name, value in self.model_dump().items():
            if isinstance(value, torch.Tensor):
                updates[name] = value.detach()
            else:
                updates[name] = value
        return self.model_copy(update=updates)

def split_experience_batch(experience: Experience) -> list[Experience]:
    """Split batched experience into individual experiences using Pydantic."""
    batch_size = experience.sequences.size(0)
    data_dict = experience.model_dump()
    split_data = [{} for _ in range(batch_size)]
    
    for key, value in data_dict.items():
        if value is None:
            for i in range(batch_size):
                split_data[i][key] = None
        else:
            vals = torch.unbind(value)
            assert batch_size == len(vals)
            for i, v in enumerate(vals):
                split_data[i][key] = v

    return [Experience.model_validate(data) for data in split_data]

def join_experience_batch(items: list[Experience]) -> Experience:
    """Join individual experiences into a batched experience."""
    if not items:
        raise ValueError("Cannot join empty experience list")
        
    batch_data = {}
    # Get keys from the first item
    keys = items[0].model_dump().keys()
    
    for key in keys:
        vals = [getattr(item, key) for item in items]
        if all(v is not None for v in vals):
            data = zero_pad_sequences(vals, "left")
        else:
            data = None
        batch_data[key] = data
        
    return Experience.model_validate(batch_data)

class ReplayBuffer:
    """Enhanced replay buffer with prioritization and sampling strategies (Pydantic-Compatible)."""
    
    def __init__(self, limit: int = 0, prioritized: bool = False, alpha: float = 0.6):
        self.limit = limit
        self.prioritized = prioritized
        self.alpha = alpha
        self.items: list[Experience] = []
        self.priorities: list[float] = []
        self.max_priority = 1.0

    def append(self, experience: Experience, priority: Optional[float] = None) -> None:
        """Add experience to buffer with optional priority."""
        items = split_experience_batch(experience)
        
        for item in items:
            self.items.append(item)
            
            if self.prioritized:
                # Ensure priority is a float before appending to self.priorities
                p = float(priority if priority is not None else self.max_priority)
                self.priorities.append(p)
                self.max_priority = max(self.max_priority, p)
            
            if self.limit > 0 and len(self.items) > self.limit:
                self.items.pop(0)
                if self.prioritized:
                    self.priorities.pop(0)

    def sample(self, batch_size: int, beta: float = 0.4) -> tuple[list[Experience], Optional[torch.Tensor]]:
        """Sample experiences from buffer with optional prioritized sampling."""
        if len(self.items) == 0:
            return [], None
        
        if not self.prioritized:
            indices = torch.randint(0, len(self.items), (min(batch_size, len(self.items)),))
            sampled_items = [self.items[i.item()] for i in indices]
            return sampled_items, None
        
        priorities = torch.tensor(self.priorities, dtype=torch.float32)
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        indices = torch.multinomial(probs, min(batch_size, len(self.items)), replacement=True)
        sampled_items = [self.items[i.item()] for i in indices]
        
        weights = (len(self.items) * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        
        return sampled_items, weights

    def update_priorities(self, indices: list[int], priorities: list[float]) -> None:
        """Update priorities for prioritized replay."""
        if not self.prioritized:
            warnings.warn("Buffer not configured for prioritized replay")
            return
        
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = float(priority)
                self.max_priority = max(self.max_priority, float(priority))

    def clear(self) -> None:
        """Clear all experiences from buffer."""
        self.items.clear()
        if self.prioritized:
            self.priorities.clear()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Experience:
        return self.items[idx]

class CircularBuffer:
    """Circular buffer for efficient memory management."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        # Use a list of Optional[Any] to satisfy typing
        self.buffer: list[Optional[Any]] = [None] * capacity
        self.position = 0
        self.size = 0

    def append(self, item: Any) -> None:
        """Add item to circular buffer."""
        self.buffer[self.position] = item
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> list[Any]:
        """Sample random items from buffer."""
        if self.size == 0:
            return []
        
        indices = torch.randint(0, self.size, (min(batch_size, self.size),))
        return [self.buffer[i.item()] for i in indices if self.buffer[i.item()] is not None]

    def get_all(self) -> list[Any]:
        """Get all items in buffer."""
        return [item for item in self.buffer if item is not None]

    def clear(self) -> None:
        """Clear buffer."""
        self.buffer = [None] * self.capacity
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

class PrioritizedExperienceReplay:
    """Prioritized Experience Replay with SumTree for efficient sampling."""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = ReplayBuffer(limit=capacity, prioritized=True, alpha=alpha)
        self.beta_increment = 0.001

    def add(self, experience: Experience, error: Optional[float] = None) -> None:
        """Add experience with priority based on TD error."""
        if error is None:
            priority = self.buffer.max_priority
        else:
            priority = float((abs(error) + 1e-6) ** self.alpha)
        
        self.buffer.append(experience, priority)

    def sample(self, batch_size: int) -> tuple[list[Experience], torch.Tensor, list[int]]:
        """Sample batch with importance sampling weights."""
        sampled_items, weights = self.buffer.sample(batch_size, self.beta)
        
        if weights is None:
            weights = torch.ones(len(sampled_items))
        
        indices = list(range(len(sampled_items)))
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return sampled_items, weights, indices

    def update_priorities(self, indices: list[int], errors: list[float]) -> None:
        """Update priorities based on new TD errors."""
        priorities = [float((abs(error) + 1e-6) ** self.alpha) for error in errors]
        self.buffer.update_priorities(indices, priorities)

def create_experience_buffer(buffer_type: str = "standard", **kwargs) -> Union[ReplayBuffer, CircularBuffer]:
    """Factory function to create experience buffer."""
    if buffer_type == "standard":
        return ReplayBuffer(**kwargs)
    elif buffer_type == "prioritized":
        return ReplayBuffer(prioritized=True, **kwargs)
    elif buffer_type == "circular":
        capacity = kwargs.get("capacity", 10000)
        return CircularBuffer(capacity)
    else:
        raise ValueError(f"Unknown buffer type: {buffer_type}")

def group_advantages(returns: torch.Tensor,
                     process_var: float = 1e-5,
                     meas_var: float = 1e-2,
                     eps: float = 1e-8) -> torch.Tensor:
    """Compute advantages using Kalman filtering."""
    from ..modules.acceleration.gpu.cuda_kernels import KalmanFilter
    
    kf = KalmanFilter(process_var, meas_var)
    flat_returns = returns.flatten()
    advantages = []

    for r in flat_returns:
        mean_est = kf.update(r.item())
        adv = r - mean_est
        advantages.append(adv)

    return torch.stack(advantages).view_as(returns)

