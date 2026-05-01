"""
TruthGPT Polyglot Core - Async/Await Support

Provides async versions of all polyglot components for high-concurrency workloads.

Features:
    - Async KV Cache with concurrent access
    - Async compression with background workers
    - Async inference with streaming support
    - WebSocket streaming interface
    - Connection pooling

Example:
    >>> import asyncio
    >>> from optimization_core.polyglot_core.async_core import (
    ...     AsyncKVCache, AsyncInferenceEngine, AsyncCompressor
    ... )
    >>> 
    >>> async def main():
    ...     cache = AsyncKVCache()
    ...     await cache.put(0, 42, data)
    ...     result = await cache.get(0, 42)
    ...     
    ...     engine = AsyncInferenceEngine()
    ...     async for token in engine.stream_generate(input_ids, forward_fn):
    ...         print(token, end="")
    >>> 
    >>> asyncio.run(main())
"""

from __future__ import annotations
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, TypeVar, Union
from dataclasses import dataclass, field
import numpy as np
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "AsyncKVCache",
    "AsyncCompressor",
    "AsyncInferenceEngine",
    "AsyncBatchScheduler",
    "WebSocketStreamer",
]


_executor = ThreadPoolExecutor(max_workers=8)


async def run_in_executor(func, *args, **kwargs):
    """Run a synchronous function in the thread pool executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, lambda: func(*args, **kwargs))


@dataclass
class AsyncKVCacheConfig:
    """Configuration for Async KV Cache."""
    max_size: int = 8192
    eviction_strategy: str = "adaptive"
    enable_compression: bool = True
    compression_threshold: int = 1024
    max_concurrent_ops: int = 100


class AsyncKVCache:
    """
    Async KV Cache with concurrent access support.
    
    Provides non-blocking cache operations suitable for async/await workflows.
    
    Args:
        config: Cache configuration
    
    Example:
        >>> cache = AsyncKVCache()
        >>> await cache.put(layer_idx=0, position=42, data=tensor_bytes)
        >>> data = await cache.get(layer_idx=0, position=42)
    """
    
    def __init__(self, config: Optional[AsyncKVCacheConfig] = None):
        self.config = config or AsyncKVCacheConfig()
        self._cache: Dict[tuple, bytes] = {}
        self._access_order: List[tuple] = []
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_ops)
        self._hit_count = 0
        self._miss_count = 0
    
    async def get(
        self,
        layer_idx: int,
        position: int,
        key: str = ""
    ) -> Optional[bytes]:
        """Get cached value asynchronously."""
        async with self._semaphore:
            cache_key = (layer_idx, position, key)
            
            async with self._lock:
                if cache_key in self._cache:
                    self._hit_count += 1
                    self._access_order.remove(cache_key)
                    self._access_order.append(cache_key)
                    return self._cache[cache_key]
                self._miss_count += 1
                return None
    
    async def put(
        self,
        layer_idx: int,
        position: int,
        data: bytes,
        key: str = ""
    ) -> None:
        """Store value in cache asynchronously."""
        async with self._semaphore:
            cache_key = (layer_idx, position, key)
            
            async with self._lock:
                if len(self._cache) >= self.config.max_size:
                    evict_key = self._access_order.pop(0)
                    del self._cache[evict_key]
                
                self._cache[cache_key] = data
                if cache_key in self._access_order:
                    self._access_order.remove(cache_key)
                self._access_order.append(cache_key)
    
    async def contains(self, layer_idx: int, position: int, key: str = "") -> bool:
        """Check if key exists asynchronously."""
        async with self._lock:
            return (layer_idx, position, key) in self._cache
    
    async def clear(self) -> None:
        """Clear all cached data asynchronously."""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._hit_count = 0
            self._miss_count = 0
    
    async def stats(self) -> Dict[str, Any]:
        """Get cache statistics asynchronously."""
        async with self._lock:
            total = self._hit_count + self._miss_count
            return {
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": self._hit_count / total if total > 0 else 0,
                "size": len(self._cache),
            }
    
    def __len__(self) -> int:
        return len(self._cache)


@dataclass
class AsyncCompressionConfig:
    """Configuration for Async Compressor."""
    algorithm: str = "lz4"
    level: int = 3
    threshold: int = 1024
    max_workers: int = 4


class AsyncCompressor:
    """
    Async Compressor with background workers.
    
    Provides non-blocking compression operations.
    
    Args:
        config: Compression configuration
    """
    
    def __init__(self, config: Optional[AsyncCompressionConfig] = None):
        self.config = config or AsyncCompressionConfig()
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
    
    async def compress(self, data: bytes) -> bytes:
        """Compress data asynchronously."""
        if len(data) < self.config.threshold:
            return data
        
        return await run_in_executor(self._sync_compress, data)
    
    async def decompress(self, data: bytes) -> bytes:
        """Decompress data asynchronously."""
        return await run_in_executor(self._sync_decompress, data)
    
    async def compress_batch(self, data_list: List[bytes]) -> List[bytes]:
        """Compress multiple data items concurrently."""
        tasks = [self.compress(data) for data in data_list]
        return await asyncio.gather(*tasks)
    
    def _sync_compress(self, data: bytes) -> bytes:
        import zlib
        return zlib.compress(data, level=self.config.level)
    
    def _sync_decompress(self, data: bytes) -> bytes:
        import zlib
        return zlib.decompress(data)


@dataclass
class AsyncInferenceConfig:
    """Configuration for Async Inference Engine."""
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    stream_chunk_size: int = 1
    timeout: float = 60.0


class AsyncInferenceEngine:
    """
    Async Inference Engine with streaming support.
    
    Provides async token generation with real-time streaming.
    
    Args:
        config: Inference configuration
    
    Example:
        >>> engine = AsyncInferenceEngine()
        >>> async for token in engine.stream_generate(input_ids, forward_fn):
        ...     print(token, end="")
    """
    
    def __init__(self, config: Optional[AsyncInferenceConfig] = None):
        self.config = config or AsyncInferenceConfig()
        np.random.seed(42)
    
    async def generate(
        self,
        input_ids: List[int],
        forward_fn: Callable[[List[int]], np.ndarray],
        max_new_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate tokens asynchronously."""
        max_tokens = max_new_tokens or self.config.max_new_tokens
        
        generated = list(input_ids)
        logprobs = []
        
        for _ in range(max_tokens):
            logits = await run_in_executor(forward_fn, generated)
            token_id = self._sample(logits)
            
            probs = self._softmax(logits / max(self.config.temperature, 1e-7))
            logprobs.append(float(np.log(probs[token_id] + 1e-10)))
            generated.append(token_id)
            
            await asyncio.sleep(0)
        
        return {
            "token_ids": generated,
            "logprobs": logprobs,
            "tokens_generated": len(generated) - len(input_ids),
        }
    
    async def stream_generate(
        self,
        input_ids: List[int],
        forward_fn: Callable[[List[int]], np.ndarray],
        max_new_tokens: Optional[int] = None
    ) -> AsyncIterator[int]:
        """
        Stream tokens as they are generated.
        
        Yields tokens one at a time for real-time streaming.
        """
        max_tokens = max_new_tokens or self.config.max_new_tokens
        
        generated = list(input_ids)
        
        for _ in range(max_tokens):
            logits = await run_in_executor(forward_fn, generated)
            token_id = self._sample(logits)
            
            generated.append(token_id)
            yield token_id
            
            await asyncio.sleep(0)
    
    def _sample(self, logits: np.ndarray) -> int:
        logits = logits / max(self.config.temperature, 1e-7)
        probs = self._softmax(logits)
        
        if self.config.top_k > 0:
            top_k_indices = np.argsort(probs)[-self.config.top_k:]
            mask = np.zeros_like(probs)
            mask[top_k_indices] = 1
            probs = probs * mask
            probs = probs / probs.sum()
        
        return int(np.random.choice(len(probs), p=probs))
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()


@dataclass
class AsyncBatchConfig:
    """Configuration for Async Batch Scheduler."""
    max_batch_size: int = 32
    max_wait_time: float = 0.1
    priority_queue: bool = True


class AsyncBatchScheduler:
    """
    Async Batch Scheduler for efficient request batching.
    
    Groups async requests for batched processing.
    """
    
    def __init__(self, config: Optional[AsyncBatchConfig] = None):
        self.config = config or AsyncBatchConfig()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._results: Dict[str, Any] = {}
        self._running = False
    
    async def add_request(
        self,
        input_ids: List[int],
        priority: int = 1
    ) -> str:
        """Add a request to the batch queue."""
        import uuid
        request_id = str(uuid.uuid4())
        await self._queue.put({
            "id": request_id,
            "input_ids": input_ids,
            "priority": priority,
        })
        return request_id
    
    async def wait_for_result(self, request_id: str, timeout: float = 60.0) -> Optional[Any]:
        """Wait for a result with timeout."""
        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < timeout:
            if request_id in self._results:
                return self._results.pop(request_id)
            await asyncio.sleep(0.01)
        return None
    
    async def get_batch(self) -> List[Dict[str, Any]]:
        """Get the next batch of requests."""
        batch = []
        try:
            while len(batch) < self.config.max_batch_size:
                req = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=self.config.max_wait_time
                )
                batch.append(req)
        except asyncio.TimeoutError:
            pass
        
        if self.config.priority_queue:
            batch.sort(key=lambda x: -x["priority"])
        
        return batch
    
    async def run(self, process_fn: Callable) -> None:
        """Run the scheduler loop."""
        self._running = True
        while self._running:
            batch = await self.get_batch()
            if batch:
                results = await process_fn(batch)
                for req, result in zip(batch, results):
                    self._results[req["id"]] = result
    
    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False


class WebSocketStreamer:
    """
    WebSocket Streaming Interface for real-time inference.
    
    Provides WebSocket server for streaming tokens to clients.
    
    Example:
        >>> streamer = WebSocketStreamer(port=8765)
        >>> await streamer.start()
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self._server = None
        self._clients: set = set()
        self._engine = AsyncInferenceEngine()
    
    async def handle_client(self, websocket, path):
        """Handle a WebSocket client connection."""
        self._clients.add(websocket)
        try:
            async for message in websocket:
                import json
                data = json.loads(message)
                
                if data.get("type") == "generate":
                    input_ids = data.get("input_ids", [])
                    max_tokens = data.get("max_tokens", 100)
                    
                    def dummy_forward(ids):
                        return np.random.randn(32000).astype(np.float32)
                    
                    async for token in self._engine.stream_generate(
                        input_ids, dummy_forward, max_tokens
                    ):
                        await websocket.send(json.dumps({
                            "type": "token",
                            "token_id": token
                        }))
                    
                    await websocket.send(json.dumps({
                        "type": "done"
                    }))
        finally:
            self._clients.discard(websocket)
    
    async def start(self):
        """Start the WebSocket server."""
        try:
            import websockets
            self._server = await websockets.serve(
                self.handle_client,
                self.host,
                self.port
            )
            logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
            await self._server.wait_closed()
        except ImportError:
            logger.warning("websockets package not available. Install with: pip install websockets")
    
    async def stop(self):
        """Stop the WebSocket server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
    
    async def broadcast(self, message: str):
        """Broadcast a message to all connected clients."""
        if self._clients:
            await asyncio.gather(
                *[client.send(message) for client in self._clients]
            )


async def example_usage():
    """Example demonstrating async components."""
    
    cache = AsyncKVCache()
    await cache.put(0, 42, b"test_data")
    data = await cache.get(0, 42)
    print(f"Cache data: {data}")
    
    compressor = AsyncCompressor()
    original = b"Hello World! " * 100
    compressed = await compressor.compress(original)
    print(f"Compression ratio: {len(compressed) / len(original):.2%}")
    
    engine = AsyncInferenceEngine()
    
    def dummy_forward(ids):
        return np.random.randn(1000).astype(np.float32)
    
    print("Generating tokens: ", end="")
    count = 0
    async for token in engine.stream_generate([1, 2, 3], dummy_forward, max_new_tokens=10):
        print(f"{token} ", end="")
        count += 1
    print(f"\nGenerated {count} tokens")


if __name__ == "__main__":
    asyncio.run(example_usage())













