"""
Streaming utilities for polyglot_core.

Provides streaming interfaces for inference and data processing.
"""

from typing import Iterator, AsyncIterator, Callable, Any, Optional, List
import numpy as np
import asyncio
from queue import Queue
from threading import Thread


class StreamProcessor:
    """
    Stream processor for polyglot_core.
    
    Provides streaming interfaces for real-time processing.
    """
    
    def __init__(self, buffer_size: int = 100):
        """
        Initialize stream processor.
        
        Args:
            buffer_size: Buffer size for streaming
        """
        self.buffer_size = buffer_size
    
    def stream_process(
        self,
        items: Iterator[Any],
        process_fn: Callable[[Any], Any],
        batch_size: int = 1
    ) -> Iterator[Any]:
        """
        Process items in a stream.
        
        Args:
            items: Iterator of items
            process_fn: Processing function
            batch_size: Batch size for processing
            
        Yields:
            Processed items
        """
        batch = []
        
        for item in items:
            batch.append(item)
            
            if len(batch) >= batch_size:
                results = process_fn(batch)
                for result in results:
                    yield result
                batch = []
        
        # Process remaining items
        if batch:
            results = process_fn(batch)
            for result in results:
                yield result
    
    async def async_stream_process(
        self,
        items: AsyncIterator[Any],
        process_fn: Callable[[Any], Any],
        batch_size: int = 1
    ) -> AsyncIterator[Any]:
        """
        Process items in an async stream.
        
        Args:
            items: Async iterator of items
            process_fn: Processing function
            batch_size: Batch size for processing
            
        Yields:
            Processed items
        """
        batch = []
        
        async for item in items:
            batch.append(item)
            
            if len(batch) >= batch_size:
                results = process_fn(batch)
                for result in results:
                    yield result
                batch = []
        
        # Process remaining items
        if batch:
            results = process_fn(batch)
            for result in results:
                yield result


class TokenStreamer:
    """
    Token streamer for text generation.
    
    Provides streaming interface for token-by-token generation.
    """
    
    def __init__(self, engine: Any, forward_fn: Callable, config: Any):
        """
        Initialize token streamer.
        
        Args:
            engine: Inference engine
            forward_fn: Forward function
            config: Generation config
        """
        self.engine = engine
        self.forward_fn = forward_fn
        self.config = config
        self._queue = Queue()
        self._thread = None
        self._stop = False
    
    def stream(self, input_ids: np.ndarray) -> Iterator[int]:
        """
        Stream tokens.
        
        Args:
            input_ids: Input token IDs
            
        Yields:
            Generated token IDs
        """
        def generate():
            try:
                result = self.engine.generate(input_ids, self.forward_fn, self.config)
                for token in result.tokens:
                    if self._stop:
                        break
                    self._queue.put(token)
                self._queue.put(None)  # End marker
            except Exception as e:
                self._queue.put(e)
        
        self._thread = Thread(target=generate)
        self._thread.start()
        
        while True:
            item = self._queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    
    def stop(self):
        """Stop streaming."""
        self._stop = True
        if self._thread:
            self._thread.join()


async def stream_tokens_async(
    engine: Any,
    input_ids: np.ndarray,
    forward_fn: Callable,
    config: Any
) -> AsyncIterator[int]:
    """
    Stream tokens asynchronously.
    
    Args:
        engine: Inference engine
        input_ids: Input token IDs
        forward_fn: Forward function
        config: Generation config
        
    Yields:
        Generated token IDs
    """
    result = engine.generate(input_ids, forward_fn, config)
    
    for token in result.tokens:
        yield token
        await asyncio.sleep(0)  # Yield control


def stream_process(
    items: Iterator[Any],
    process_fn: Callable[[Any], Any],
    batch_size: int = 1
) -> Iterator[Any]:
    """Convenience function to stream process items."""
    processor = StreamProcessor()
    return processor.stream_process(items, process_fn, batch_size)













