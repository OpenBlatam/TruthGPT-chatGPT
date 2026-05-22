"""
Performance and stress tests for the TruthGPT system.
Tests system behavior under load and with large datasets.
"""
import numpy as np
import pytest
import sys
import os
import time
import psutil
import gc
from unittest.mock import patch
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add paths for imports
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Automate-Inteligence', 'AI-Automate', 'Transformers'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Automate-Inteligence', 'AI-Automate', 'MAIN-Architecture', 'NLP'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Automate-Inteligence', 'AI-Automate', 'MAIN-Architecture'))

from transformer import attention
from manager import CommandManager
from production import execute_action_based_on_text
from finetune import FineTuner, softmax


class TestPerformanceBenchmarks:
    """Performance benchmarks for core functions."""
    
    def test_softmax_performance_scaling(self):
        """Test softmax performance with increasing input sizes."""
        sizes = [10, 100, 1000, 5000]
        times = []
        
        for size in sizes:
            x = np.random.rand(size)
            
            start_time = time.perf_counter()
            for _ in range(100):  # Multiple iterations for better measurement
                result = softmax(x)
            end_time = time.perf_counter()
            
            avg_time = (end_time - start_time) / 100
            times.append(avg_time)
            
            # Verify correctness isn't compromised for performance
            assert np.isclose(np.sum(result), 1.0)
            assert np.all(result > 0)
        
        # Performance should scale reasonably (not exponentially)
        for i in range(1, len(times)):
            ratio = times[i] / times[i-1]
            size_ratio = sizes[i] / sizes[i-1]
            # Time ratio should be roughly proportional to size ratio
            assert ratio < size_ratio * 2, f"Performance degradation too severe: {ratio} vs {size_ratio}"
    
    def test_attention_performance_scaling(self):
        """Test attention mechanism performance with increasing matrix sizes."""
        sizes = [4, 8, 16, 32]
        times = []
        
        for size in sizes:
            Q = np.random.rand(size, size)
            K = np.random.rand(size, size)
            V = np.random.rand(size, size)
            
            start_time = time.perf_counter()
            for _ in range(10):  # Fewer iterations for larger matrices
                result = attention(Q, K, V, dk=size)
            end_time = time.perf_counter()
            
            avg_time = (end_time - start_time) / 10
            times.append(avg_time)
            
            # Verify correctness
            assert result.shape == (size, size)
            assert np.all(np.isfinite(result))
        
        # Check that performance scales reasonably
        for i in range(1, len(times)):
            ratio = times[i] / times[i-1]
            size_ratio = sizes[i] / sizes[i-1]
            # Attention is O(n^2) in sequence length, so expect quadratic scaling
            expected_ratio = size_ratio ** 2
            assert ratio < expected_ratio * 3, f"Performance worse than expected: {ratio} vs {expected_ratio}"
    
    def test_finetuner_performance_with_large_datasets(self):
        """Test FineTuner performance with large datasets."""
        sizes = [(10, 10), (50, 50), (100, 100)]
        times = []
        
        for rows, cols in sizes:
            data = np.random.rand(rows, cols)
            tuner = FineTuner()
            
            start_time = time.perf_counter()
            result = tuner.finetune(n=100, data=data, learning_rate=0.01)
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
            
            # Verify correctness
            assert result['status'] == 'completed'
            if result['softmax_result'] is not None:
                assert result['softmax_result'].shape == (rows, cols)
        
        # Performance should scale reasonably
        assert all(t < 5.0 for t in times), "FineTuner should complete in reasonable time"
    
    def test_command_manager_performance_with_many_commands(self):
        """Test CommandManager performance with large numbers of commands."""
        command_counts = [10, 100, 1000]
        times = []
        
        for count in command_counts:
            manager = CommandManager()
            
            # Create simple commands
            class SimpleCommand:
                def __init__(self, cmd_id):
                    self.cmd_id = cmd_id
                    self.executed = False
                
                def execute(self):
                    self.executed = True
                
                def undo(self):
                    self.executed = False
            
            # Add commands
            commands = []
            for i in range(count):
                cmd = SimpleCommand(i)
                commands.append(cmd)
                manager.add_command(cmd)
            
            # Measure execution time
            start_time = time.perf_counter()
            manager.execute_commands()
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
            
            # Verify all commands were executed
            assert all(cmd.executed for cmd in commands)
        
        # Performance should scale linearly
        for i in range(1, len(times)):
            ratio = times[i] / times[i-1]
            count_ratio = command_counts[i] / command_counts[i-1]
            assert ratio < count_ratio * 2, f"Command execution scaling worse than linear: {ratio} vs {count_ratio}"


class TestMemoryUsage:
    """Test memory usage and potential memory leaks."""
    
    def test_softmax_memory_usage(self):
        """Test that softmax doesn't leak memory with repeated calls."""
        initial_memory = psutil.Process().memory_info().rss
        
        # Perform many softmax operations
        for _ in range(1000):
            x = np.random.rand(1000)
            result = softmax(x)
            del result  # Explicit cleanup
        
        gc.collect()  # Force garbage collection
        final_memory = psutil.Process().memory_info().rss
        
        # Memory increase should be minimal
        memory_increase = final_memory - initial_memory
        assert memory_increase < 50 * 1024 * 1024, f"Memory increase too large: {memory_increase / 1024 / 1024:.2f} MB"
    
    def test_attention_memory_efficiency(self):
        """Test attention mechanism memory efficiency."""
        initial_memory = psutil.Process().memory_info().rss
        
        # Perform attention operations with moderately large matrices
        for _ in range(100):
            size = 64
            Q = np.random.rand(size, size)
            K = np.random.rand(size, size)
            V = np.random.rand(size, size)
            
            result = attention(Q, K, V, dk=size)
            
            # Clean up explicitly
            del Q, K, V, result
        
        gc.collect()
        final_memory = psutil.Process().memory_info().rss
        
        memory_increase = final_memory - initial_memory
        assert memory_increase < 100 * 1024 * 1024, f"Memory increase too large: {memory_increase / 1024 / 1024:.2f} MB"
    
    def test_finetuner_memory_management(self):
        """Test FineTuner memory management with large datasets."""
        initial_memory = psutil.Process().memory_info().rss
        
        # Create and destroy multiple FineTuner instances
        for _ in range(50):
            tuner = FineTuner()
            data = np.random.rand(100, 100)
            result = tuner.finetune(n=50, data=data)
            
            # Clean up
            del tuner, data, result
        
        gc.collect()
        final_memory = psutil.Process().memory_info().rss
        
        memory_increase = final_memory - initial_memory
        assert memory_increase < 200 * 1024 * 1024, f"Memory increase too large: {memory_increase / 1024 / 1024:.2f} MB"


class TestStressTests:
    """Stress tests to verify system stability under extreme conditions."""
    
    def test_softmax_with_extreme_values(self):
        """Test softmax stability with extreme input values."""
        extreme_cases = [
            np.array([1e10, 1e10, 1e10]),  # Very large values
            np.array([-1e10, -1e10, -1e10]),  # Very negative values
            np.array([1e10, -1e10, 0]),  # Mixed extreme values
            np.array([1e-10, 1e-10, 1e-10]),  # Very small values
            np.array([0, 0, 0]),  # All zeros
        ]
        
        for case in extreme_cases:
            result = softmax(case)
            
            # Should still produce valid probability distribution
            assert np.isclose(np.sum(result), 1.0, rtol=1e-10)
            assert np.all(result >= 0)
            assert np.all(np.isfinite(result))
    
    def test_attention_with_extreme_matrices(self):
        """Test attention mechanism with extreme matrix values."""
        size = 8
        
        extreme_cases = [
            (np.ones((size, size)) * 1e6, np.ones((size, size)) * 1e6, np.random.rand(size, size)),
            (np.ones((size, size)) * -1e6, np.ones((size, size)) * -1e6, np.random.rand(size, size)),
            (np.zeros((size, size)), np.zeros((size, size)), np.random.rand(size, size)),
            (np.random.rand(size, size), np.random.rand(size, size), np.ones((size, size)) * 1e6),
        ]
        
        for Q, K, V in extreme_cases:
            result = attention(Q, K, V, dk=size)
            
            # Should produce finite output
            assert np.all(np.isfinite(result))
            assert result.shape == (size, size)
    
    def test_finetuner_with_extreme_parameters(self):
        """Test FineTuner with extreme parameter values."""
        tuner = FineTuner()
        data = np.random.rand(10, 10)
        
        extreme_cases = [
            {'n': 1, 'learning_rate': 1e-10},  # Very small learning rate
            {'n': 1, 'learning_rate': 0.99},   # Very large learning rate
            {'n': 10000, 'learning_rate': 0.01},  # Very large n
        ]
        
        for params in extreme_cases:
            result = tuner.finetune(data=data, **params)
            
            # Should complete without crashing
            assert result['status'] == 'completed'
            assert result['n'] == params['n']
            assert result['learning_rate'] == params['learning_rate']
    
    def test_command_manager_with_failing_commands(self):
        """Test CommandManager resilience with commands that fail."""
        manager = CommandManager()
        
        class FailingCommand:
            def __init__(self, should_fail=False):
                self.should_fail = should_fail
                self.executed = False
            
            def execute(self):
                if self.should_fail:
                    raise RuntimeError("Simulated command failure")
                self.executed = True
            
            def undo(self):
                pass
        
        # Add mix of failing and successful commands
        commands = []
        for i in range(100):
            cmd = FailingCommand(should_fail=(i % 10 == 0))  # Every 10th command fails
            commands.append(cmd)
            manager.add_command(cmd)
        
        # Execute commands - should handle failures gracefully
        manager.execute_commands()
        
        # Successful commands should have executed
        successful_commands = [cmd for cmd in commands if not cmd.should_fail]
        failed_commands = [cmd for cmd in commands if cmd.should_fail]
        
        assert all(cmd.executed for cmd in successful_commands)
        assert all(not cmd.executed for cmd in failed_commands)


class TestConcurrencyStress:
    """Stress tests for concurrent operations."""
    
    def test_concurrent_softmax_operations(self):
        """Test many concurrent softmax operations."""
        def worker(worker_id):
            results = []
            for i in range(100):
                x = np.random.rand(100) + worker_id  # Different data per worker
                result = softmax(x)
                results.append(np.sum(result))  # Should always be 1.0
            return results
        
        # Run multiple workers concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, i) for i in range(4)]
            
            all_results = []
            for future in as_completed(futures):
                results = future.result()
                all_results.extend(results)
        
        # All results should be valid
        assert len(all_results) == 400  # 4 workers * 100 operations
        assert all(np.isclose(result, 1.0, rtol=1e-10) for result in all_results)
    
    def test_concurrent_attention_operations(self):
        """Test concurrent attention operations."""
        def worker(worker_id):
            results = []
            for i in range(20):
                size = 8
                Q = np.random.rand(size, size) + worker_id
                K = np.random.rand(size, size) + worker_id
                V = np.random.rand(size, size) + worker_id
                
                result = attention(Q, K, V, dk=size)
                results.append(np.all(np.isfinite(result)))
            return results
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, i) for i in range(4)]
            
            all_results = []
            for future in as_completed(futures):
                results = future.result()
                all_results.extend(results)
        
        # All operations should produce finite results
        assert len(all_results) == 80  # 4 workers * 20 operations
        assert all(result for result in all_results)
    
    def test_concurrent_finetuner_operations(self):
        """Test concurrent FineTuner operations."""
        def worker(worker_id):
            tuner = FineTuner()
            data = np.random.rand(20, 20) + worker_id
            
            result = tuner.finetune(n=100, data=data, learning_rate=0.01)
            return result['status'] == 'completed'
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(worker, i) for i in range(3)]
            
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
        
        # All fine-tuning operations should complete successfully
        assert len(results) == 3
        assert all(results)


class TestLongRunningOperations:
    """Test system behavior during long-running operations."""
    
    def test_extended_finetuning_session(self):
        """Test extended fine-tuning session."""
        tuner = FineTuner()
        data = np.random.rand(50, 50)
        
        # Run extended fine-tuning
        start_time = time.time()
        result = tuner.finetune(n=5000, data=data, learning_rate=0.001)
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 30.0, "Extended fine-tuning took too long"
        assert result['status'] == 'completed'
        assert result['n'] == 5000
    
    def test_repeated_attention_operations(self):
        """Test repeated attention operations over time."""
        size = 16
        start_time = time.time()
        
        for i in range(1000):
            Q = np.random.rand(size, size)
            K = np.random.rand(size, size)
            V = np.random.rand(size, size)
            
            result = attention(Q, K, V, dk=size)
            
            # Verify correctness throughout
            assert result.shape == (size, size)
            assert np.all(np.isfinite(result))
            
            # Check for performance degradation
            if i % 100 == 0 and i > 0:
                current_time = time.time()
                elapsed = current_time - start_time
                rate = i / elapsed
                assert rate > 10, f"Performance degraded: {rate:.2f} ops/sec at iteration {i}"
    
    def test_command_manager_long_session(self):
        """Test CommandManager during long session with many operations."""
        manager = CommandManager()
        
        class TimestampCommand:
            def __init__(self, timestamp):
                self.timestamp = timestamp
                self.executed = False
            
            def execute(self):
                self.executed = True
            
            def undo(self):
                self.executed = False
        
        start_time = time.time()
        
        # Add and execute commands over time
        for i in range(2000):
            cmd = TimestampCommand(time.time())
            manager.add_command(cmd)
            
            if i % 100 == 0:
                manager.execute_commands()
                
                # Test undo operations periodically
                if i % 500 == 0 and i > 0:
                    for _ in range(10):
                        manager.undo_last_command()
        
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 10.0, "Long session took too long"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])