"""
Model profiling utilities.
"""
import logging
import torch
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelProfiler:
    """
    Profiler for analyzing model performance.
    """
    
    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize profiler.
        
        Args:
            log_dir: Directory for profile logs
        """
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def profile(
        self,
        model: torch.nn.Module,
        input_shape: tuple,
        num_runs: int = 10,
        warmup_runs: int = 3,
        activities: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        Profile model performance.
        
        Args:
            model: Model to profile
            input_shape: Input tensor shape
            num_runs: Number of profiling runs
            warmup_runs: Number of warmup runs
            activities: Profiling activities
        
        Returns:
            Dictionary with profiling results
        """
        if activities is None:
            activities = [
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        
        # Create dummy input
        dummy_input = torch.zeros(input_shape)
        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda()
            model = model.cuda()
        
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(dummy_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Profile
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = model(dummy_input)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Process results
        if self.log_dir:
            prof.export_chrome_trace(str(self.log_dir / "trace.json"))
        
        # Get statistics
        key_averages = prof.key_averages()
        
        stats = {
            "cpu_time_total": sum(e.cpu_time_total for e in key_averages),
            "cuda_time_total": sum(e.cuda_time_total for e in key_averages) if torch.cuda.is_available() else 0,
            "cpu_memory_usage": sum(e.cpu_memory_usage for e in key_averages),
            "cuda_memory_usage": sum(e.cuda_memory_usage for e in key_averages) if torch.cuda.is_available() else 0,
        }
        
        logger.info("Model profiling completed")
        return stats



