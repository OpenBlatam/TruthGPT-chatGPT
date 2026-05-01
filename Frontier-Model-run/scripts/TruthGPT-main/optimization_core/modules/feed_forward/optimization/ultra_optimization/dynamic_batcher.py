"""
Dynamic Batching
Ultra-fast dynamic batching with intelligent batch sizing, pipeline optimization, and load balancing.
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
import threading
import queue
import heapq
from collections import deque
import statistics

class BatchItem:
    """Individual item in a batch."""
    
    def __init__(self, data: torch.Tensor, priority: float = 1.0, timestamp: float = None):
        self.data = data
        self.priority = priority
        self.timestamp = timestamp or time.time()
        self.batch_id = None
        self.processing_time = 0.0
    
    def __lt__(self, other):
        """Comparison for priority queue."""
        return self.priority > other.priority  # Higher priority first

class DynamicBatch:
    """Dynamic batch with intelligent sizing."""
    
    def __init__(self, max_size: int, max_wait_time: float = 0.1):
        self.max_size = max_size
        self.max_wait_time = max_wait_time
        self.items = []
        self.creation_time = time.time()
        self.is_full = False
        self.is_timed_out = False
    
    def add_item(self, item: BatchItem) -> bool:
        """Add item to batch."""
        if len(self.items) >= self.max_size:
            return False
        
        self.items.append(item)
        item.batch_id = id(self)
        
        # Check if batch is full
        if len(self.items) >= self.max_size:
            self.is_full = True
        
        return True
    
    def is_ready(self) -> bool:
        """Check if batch is ready for processing."""
        current_time = time.time()
        
        # Check if batch is full
        if self.is_full:
            return True
        
        # Check if batch has timed out
        if current_time - self.creation_time >= self.max_wait_time:
            self.is_timed_out = True
            return True
        
        return False
    
    def get_batch_tensor(self) -> torch.Tensor:
        """Get batch as tensor."""
        if not self.items:
            return torch.empty(0)
        
        # Stack tensors
        tensors = [item.data for item in self.items]
        return torch.stack(tensors)
    
    def get_priorities(self) -> List[float]:
        """Get priorities of items in batch."""
        return [item.priority for item in self.items]
    
    def get_processing_time(self) -> float:
        """Get total processing time for batch."""
        return sum(item.processing_time for item in self.items)

class BatchScheduler:
    """Intelligent batch scheduler."""
    
    def __init__(self, config: 'BatchingConfig'):
        self.config = config
        self.pending_items = []
        self.active_batches = []
        self.completed_batches = []
        self.scheduler_stats = {
            'total_batches': 0,
            'total_items': 0,
            'average_batch_size': 0.0,
            'average_wait_time': 0.0,
            'throughput': 0.0
        }
    
    def add_item(self, item: BatchItem) -> None:
        """Add item to scheduler."""
        self.pending_items.append(item)
        self.scheduler_stats['total_items'] += 1
    
    def get_next_batch(self) -> Optional[DynamicBatch]:
        """Get next batch for processing."""
        if not self.pending_items:
            return None
        
        # Create new batch
        batch = DynamicBatch(
            max_size=self.config.max_batch_size,
            max_wait_time=self.config.max_wait_time
        )
        
        # Add items to batch
        items_to_remove = []
        for i, item in enumerate(self.pending_items):
            if batch.add_item(item):
                items_to_remove.append(i)
            
            if batch.is_ready():
                break
        
        # Remove items from pending list
        for i in reversed(items_to_remove):
            self.pending_items.pop(i)
        
        # Add to active batches
        self.active_batches.append(batch)
        self.scheduler_stats['total_batches'] += 1
        
        return batch
    
    def complete_batch(self, batch: DynamicBatch) -> None:
        """Mark batch as completed."""
        if batch in self.active_batches:
            self.active_batches.remove(batch)
            self.completed_batches.append(batch)
            
            # Update statistics
            self._update_statistics(batch)
    
    def _update_statistics(self, batch: DynamicBatch) -> None:
        """Update scheduler statistics."""
        batch_size = len(batch.items)
        wait_time = time.time() - batch.creation_time
        
        # Update average batch size
        total_batches = self.scheduler_stats['total_batches']
        current_avg = self.scheduler_stats['average_batch_size']
        self.scheduler_stats['average_batch_size'] = (current_avg * (total_batches - 1) + batch_size) / total_batches
        
        # Update average wait time
        current_avg_wait = self.scheduler_stats['average_wait_time']
        self.scheduler_stats['average_wait_time'] = (current_avg_wait * (total_batches - 1) + wait_time) / total_batches
        
        # Update throughput
        total_items = self.scheduler_stats['total_items']
        total_time = time.time() - self.scheduler_stats.get('start_time', time.time())
        if total_time > 0:
            self.scheduler_stats['throughput'] = total_items / total_time
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return self.scheduler_stats.copy()

class LoadBalancer:
    """Load balancer for batch processing."""
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.worker_loads = [0.0] * num_workers
        self.worker_queues = [queue.Queue() for _ in range(num_workers)]
        self.worker_stats = [{'processed': 0, 'total_time': 0.0} for _ in range(num_workers)]
    
    def assign_batch(self, batch: DynamicBatch) -> int:
        """Assign batch to least loaded worker."""
        # Find least loaded worker
        min_load = min(self.worker_loads)
        worker_id = self.worker_loads.index(min_load)
        
        # Assign batch to worker
        self.worker_queues[worker_id].put(batch)
        
        # Update load
        batch_size = len(batch.items)
        self.worker_loads[worker_id] += batch_size
        
        return worker_id
    
    def complete_batch(self, worker_id: int, processing_time: float) -> None:
        """Mark batch as completed for worker."""
        batch_size = len(self.worker_queues[worker_id].get().items)
        self.worker_loads[worker_id] -= batch_size
        
        # Update worker statistics
        self.worker_stats[worker_id]['processed'] += 1
        self.worker_stats[worker_id]['total_time'] += processing_time
    
    def get_worker_stats(self) -> List[Dict[str, Any]]:
        """Get worker statistics."""
        stats = []
        for i, worker_stat in enumerate(self.worker_stats):
            stats.append({
                'worker_id': i,
                'load': self.worker_loads[i],
                'processed': worker_stat['processed'],
                'total_time': worker_stat['total_time'],
                'average_time': worker_stat['total_time'] / max(worker_stat['processed'], 1)
            })
        return stats

class PipelineOptimizer:
    """Pipeline optimizer for batch processing."""
    
    def __init__(self, config: 'BatchingConfig'):
        self.config = config
        self.pipeline_stages = []
        self.stage_stats = []
        self.optimization_history = []
    
    def add_stage(self, stage: Callable, name: str) -> None:
        """Add pipeline stage."""
        self.pipeline_stages.append({
            'name': name,
            'function': stage,
            'stats': {'calls': 0, 'total_time': 0.0, 'average_time': 0.0}
        })
    
    def process_batch(self, batch: DynamicBatch) -> Any:
        """Process batch through pipeline."""
        result = batch
        
        for stage in self.pipeline_stages:
            start_time = time.time()
            
            # Process stage
            result = stage['function'](result)
            
            # Update statistics
            stage_time = time.time() - start_time
            stage['stats']['calls'] += 1
            stage['stats']['total_time'] += stage_time
            stage['stats']['average_time'] = stage['stats']['total_time'] / stage['stats']['calls']
        
        return result
    
    def optimize_pipeline(self) -> None:
        """Optimize pipeline based on statistics."""
        # Analyze stage performance
        stage_times = [stage['stats']['average_time'] for stage in self.pipeline_stages]
        
        # Find bottlenecks
        max_time = max(stage_times)
        bottleneck_stages = [i for i, time in enumerate(stage_times) if time >= max_time * 0.8]
        
        # Record optimization
        optimization = {
            'timestamp': time.time(),
            'bottleneck_stages': bottleneck_stages,
            'stage_times': stage_times,
            'total_time': sum(stage_times)
        }
        self.optimization_history.append(optimization)
    
    def get_pipeline_stats(self) -> List[Dict[str, Any]]:
        """Get pipeline statistics."""
        stats = []
        for stage in self.pipeline_stages:
            stats.append({
                'name': stage['name'],
                'calls': stage['stats']['calls'],
                'total_time': stage['stats']['total_time'],
                'average_time': stage['stats']['average_time']
            })
        return stats

from .models import BatchingConfig

class DynamicBatcher:
    """
    Dynamic batcher for ultra-fast batch processing.
    """
    
    def __init__(self, config: BatchingConfig):
        self.config = config
        self.logger = logging.getLogger(f'{self.__class__.__name__}')
        self.scheduler = BatchScheduler(config)
        self.load_balancer = LoadBalancer(config.num_workers)
        self.pipeline_optimizer = PipelineOptimizer(config)
        self.batching_stats = {
            'total_batches': 0,
            'total_items': 0,
            'average_batch_size': 0.0,
            'average_wait_time': 0.0,
            'throughput': 0.0,
            'efficiency': 0.0
        }
        
        # Start background processing
        self._start_background_processing()
    
    def _start_background_processing(self) -> None:
        """Start background processing thread."""
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
    
    def _processing_loop(self) -> None:
        """Background processing loop."""
        while True:
            try:
                # Get next batch
                batch = self.scheduler.get_next_batch()
                if batch:
                    # Process batch
                    self._process_batch(batch)
                else:
                    # No batches available, sleep briefly
                    time.sleep(0.001)
            except Exception as e:
                self.logger.error(f"Background processing error: {e}")
                time.sleep(0.1)
    
    def add_item(self, data: torch.Tensor, priority: float = 1.0) -> None:
        """Add item for batching."""
        item = BatchItem(data, priority)
        self.scheduler.add_item(item)
        self.batching_stats['total_items'] += 1
    
    def _process_batch(self, batch: DynamicBatch) -> None:
        """Process batch through pipeline."""
        start_time = time.time()
        
        # Assign to worker
        worker_id = self.load_balancer.assign_batch(batch)
        
        # Process through pipeline
        result = self.pipeline_optimizer.process_batch(batch)
        
        # Update statistics
        processing_time = time.time() - start_time
        self.load_balancer.complete_batch(worker_id, processing_time)
        self.scheduler.complete_batch(batch)
        
        # Update batching statistics
        self.batching_stats['total_batches'] += 1
        self._update_batching_statistics(batch, processing_time)
    
    def _update_batching_statistics(self, batch: DynamicBatch, processing_time: float) -> None:
        """Update batching statistics."""
        batch_size = len(batch.items)
        wait_time = time.time() - batch.creation_time
        
        # Update average batch size
        total_batches = self.batching_stats['total_batches']
        current_avg = self.batching_stats['average_batch_size']
        self.batching_stats['average_batch_size'] = (current_avg * (total_batches - 1) + batch_size) / total_batches
        
        # Update average wait time
        current_avg_wait = self.batching_stats['average_wait_time']
        self.batching_stats['average_wait_time'] = (current_avg_wait * (total_batches - 1) + wait_time) / total_batches
        
        # Update throughput
        total_items = self.batching_stats['total_items']
        total_time = time.time() - self.batching_stats.get('start_time', time.time())
        if total_time > 0:
            self.batching_stats['throughput'] = total_items / total_time
        
        # Update efficiency
        if batch_size > 0:
            efficiency = batch_size / (batch_size + wait_time * 1000)  # Simplified efficiency metric
            self.batching_stats['efficiency'] = efficiency
    
    def get_batching_stats(self) -> Dict[str, Any]:
        """Get batching statistics."""
        return {
            'batching_stats': self.batching_stats.copy(),
            'scheduler_stats': self.scheduler.get_statistics(),
            'worker_stats': self.load_balancer.get_worker_stats(),
            'pipeline_stats': self.pipeline_optimizer.get_pipeline_stats()
        }
    
    def optimize_batching(self) -> None:
        """Optimize batching configuration."""
        # Analyze current performance
        stats = self.get_batching_stats()
        
        # Optimize batch size
        if self.config.enable_adaptive_batching:
            self._optimize_batch_size(stats)
        
        # Optimize pipeline
        if self.config.enable_pipeline_optimization:
            self.pipeline_optimizer.optimize_pipeline()
        
        # Optimize load balancing
        if self.config.enable_load_balancing:
            self._optimize_load_balancing(stats)
    
    def _optimize_batch_size(self, stats: Dict[str, Any]) -> None:
        """Optimize batch size based on performance."""
        current_throughput = stats['batching_stats']['throughput']
        current_efficiency = stats['batching_stats']['efficiency']
        
        # Adjust batch size based on performance
        if current_efficiency < 0.5:
            # Increase batch size
            self.config.max_batch_size = min(self.config.max_batch_size * 1.2, 64)
        elif current_efficiency > 0.8:
            # Decrease batch size
            self.config.max_batch_size = max(self.config.max_batch_size * 0.9, 1)
    
    def _optimize_load_balancing(self, stats: Dict[str, Any]) -> None:
        """Optimize load balancing."""
        worker_stats = stats['worker_stats']
        
        # Check for load imbalance
        loads = [worker['load'] for worker in worker_stats]
        max_load = max(loads)
        min_load = min(loads)
        
        if max_load - min_load > 5:  # Significant imbalance
            # Rebalance loads
            self.logger.info("Load imbalance detected, rebalancing...")
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        # Stop background processing
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=1.0)
        
        self.logger.info("Dynamic batcher cleanup completed")





