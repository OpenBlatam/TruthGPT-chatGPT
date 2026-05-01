"""
GPU Streaming Management
"""
import torch
import logging

from .config import GPUAcceleratorConfig

logger = logging.getLogger(__name__)

class GPUStreamManager:
    """Manage multiple CUDA streams for concurrent operations."""
    
    def __init__(self, config: GPUAcceleratorConfig):
        self.config = config
        self.streams = []
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create streams
        if self.config.device == "cuda" and torch.cuda.is_available():
            self._create_streams()
            self.logger.info(f"✅ GPU Stream Manager initialized with {len(self.streams)} streams")
        else:
            self.logger.info("GPU Stream Manager initialized (CPU mode, streams inactive)")
    
    def _create_streams(self):
        """Create CUDA streams."""
        num_streams = getattr(self.config, 'num_streams', 4)
        
        for i in range(num_streams):
            stream = torch.cuda.Stream(
                device=self.config.device_id,
                priority=-1 if i == 0 else 0
            )
            self.streams.append(stream)
    
    def get_stream(self, index: int = 0) -> torch.cuda.Stream | None:
        """Get a CUDA stream by index."""
        if not self.streams:
            return None
        if 0 <= index < len(self.streams):
            return self.streams[index]
        return self.streams[0]
    
    def synchronize_all(self):
        """Synchronize all streams."""
        for stream in self.streams:
            stream.synchronize()
    
    def synchronize_stream(self, index: int):
        """Synchronize a specific stream."""
        if 0 <= index < len(self.streams):
            self.streams[index].synchronize()

