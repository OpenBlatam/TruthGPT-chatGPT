"""
Unified Data Loader Interface

Provides Python interface to Rust and Go data loading backends.
"""
from typing import Optional, List, Dict, Any, Iterator, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    from truthgpt_rust import PyDataLoader
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

class DataLoader:
    """
    Unified data loader interface.
    
    Automatically selects best available backend:
    1. Rust (fastest, parallel)
    2. Python fallback (sequential)
    """
    
    def __init__(
        self,
        file_path: Union[str, Path],
        batch_size: int = 32,
        num_workers: int = 4,
        backend: Optional[str] = None,
        **kwargs
    ):
        self.file_path = Path(file_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        if backend is None:
            backend = "rust" if RUST_AVAILABLE else "python"
        
        self.backend = backend
        self._loader = None
        self._setup_backend(**kwargs)
    
    def _setup_backend(self, **kwargs):
        """Setup backend implementation."""
        if self.backend == "rust" and RUST_AVAILABLE:
            try:
                self._loader = PyDataLoader(
                    file_path=str(self.file_path),
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    **kwargs
                )
                logger.info("Rust data loader initialized")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize Rust loader: {e}")
        
        self._loader = None
        logger.info("Python data loader initialized")
    
    def __iter__(self) -> Iterator[List[Dict[str, Any]]]:
        """Iterate over batches."""
        if self.backend == "rust" and self._loader:
            try:
                for batch in self._loader.iter_batches():
                    yield batch
                return
            except Exception as e:
                logger.warning(f"Rust iteration failed: {e}, falling back")
        
        yield from self._iter_python()
    
    def _iter_python(self) -> Iterator[List[Dict[str, Any]]]:
        """Python fallback iteration."""
        import json
        
        batch = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    batch.append(data)
                    
                    if len(batch) >= self.batch_size:
                        yield batch
                        batch = []
                except json.JSONDecodeError:
                    continue
            
            if batch:
                yield batch
    
    def load_all(self) -> List[Dict[str, Any]]:
        """Load all data into memory."""
        all_data = []
        for batch in self:
            all_data.extend(batch)
        return all_data
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics."""
        stats = {
            "file_path": str(self.file_path),
            "backend": self.backend,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
        }
        
        if self.backend == "rust" and self._loader:
            try:
                rust_stats = self._loader.stats()
                stats.update(rust_stats)
            except Exception:
                pass
        
        return stats

def create_data_loader(
    file_path: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """Factory function to create data loader."""
    return DataLoader(
        file_path=file_path,
        batch_size=batch_size,
        num_workers=num_workers,
        **kwargs
    )

__all__ = ["DataLoader", "create_data_loader"]













