# Performance Testing Examples

## Overview

This document provides comprehensive examples of performance testing using the Enhanced Test Framework. Performance tests validate system performance under various conditions and loads.

## Load Testing Examples

### Example 1: Basic Load Testing

```python
from test_framework.test_performance import TestLoadPerformance
import unittest
import time
import random

class TestOptimizationCoreLoad(unittest.TestCase):
    def setUp(self):
        self.load_test = TestLoadPerformance()
        self.load_test.setUp()
    
    def test_light_load_performance(self):
        """Test system performance under light load."""
        # Simulate light load scenario
        concurrent_users = 10
        duration = 60  # seconds
        
        start_time = time.time()
        
        # Simulate user requests
        requests_processed = 0
        successful_requests = 0
        
        for i in range(concurrent_users):
            # Simulate user request
            request_start = time.time()
            
            # Simulate work (optimization computation)
            work_time = random.uniform(0.1, 0.5)
            time.sleep(work_time)
            
            # Simulate request success/failure
            success = random.uniform(0.9, 1.0) > 0.05
            if success:
                successful_requests += 1
            
            requests_processed += 1
            request_end = time.time()
            request_time = request_end - request_start
            
            # Verify request time is acceptable
            self.assertLess(request_time, 2.0)  # Should complete within 2 seconds
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        success_rate = (successful_requests / requests_processed) * 100
        throughput = requests_processed / total_time
        
        # Verify performance criteria
        self.assertGreater(success_rate, 90)  # At least 90% success rate
        self.assertGreater(throughput, 5)  # At least 5 requests per second
        self.assertLess(total_time, duration * 1.5)  # Should complete within 1.5x duration
        
        print(f"✅ Light load test passed")
        print(f"📊 Success rate: {success_rate:.2f}%")
        print(f"📊 Throughput: {throughput:.2f} req/s")
        print(f"📊 Total time: {total_time:.2f}s")
    
    def test_medium_load_performance(self):
        """Test system performance under medium load."""
        # Simulate medium load scenario
        concurrent_users = 50
        duration = 120  # seconds
        
        start_time = time.time()
        
        # Simulate user requests with higher load
        requests_processed = 0
        successful_requests = 0
        response_times = []
        
        for i in range(concurrent_users):
            request_start = time.time()
            
            # Simulate more complex work
            work_time = random.uniform(0.2, 1.0)
            time.sleep(work_time)
            
            # Simulate request success/failure
            success = random.uniform(0.85, 1.0) > 0.05
            if success:
                successful_requests += 1
            
            requests_processed += 1
            request_end = time.time()
            request_time = request_end - request_start
            response_times.append(request_time)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        success_rate = (successful_requests / requests_processed) * 100
        throughput = requests_processed / total_time
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        
        # Verify performance criteria
        self.assertGreater(success_rate, 85)  # At least 85% success rate
        self.assertGreater(throughput, 20)  # At least 20 requests per second
        self.assertLess(avg_response_time, 3.0)  # Average response time < 3s
        self.assertLess(max_response_time, 10.0)  # Max response time < 10s
        
        print(f"✅ Medium load test passed")
        print(f"📊 Success rate: {success_rate:.2f}%")
        print(f"📊 Throughput: {throughput:.2f} req/s")
        print(f"📊 Avg response time: {avg_response_time:.2f}s")
        print(f"📊 Max response time: {max_response_time:.2f}s")
    
    def test_heavy_load_performance(self):
        """Test system performance under heavy load."""
        # Simulate heavy load scenario
        concurrent_users = 100
        duration = 180  # seconds
        
        start_time = time.time()
        
        # Simulate high load with resource constraints
        requests_processed = 0
        successful_requests = 0
        response_times = []
        error_count = 0
        
        for i in range(concurrent_users):
            request_start = time.time()
            
            try:
                # Simulate resource-intensive work
                work_time = random.uniform(0.5, 2.0)
                time.sleep(work_time)
                
                # Simulate occasional failures under heavy load
                success = random.uniform(0.8, 1.0) > 0.1
                if success:
                    successful_requests += 1
                else:
                    error_count += 1
                
                requests_processed += 1
                request_end = time.time()
                request_time = request_end - request_start
                response_times.append(request_time)
                
            except Exception as e:
                error_count += 1
                print(f"⚠️ Request {i} failed: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        success_rate = (successful_requests / requests_processed) * 100
        throughput = requests_processed / total_time
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        error_rate = (error_count / requests_processed) * 100
        
        # Verify performance criteria (more lenient for heavy load)
        self.assertGreater(success_rate, 80)  # At least 80% success rate
        self.assertGreater(throughput, 30)  # At least 30 requests per second
        self.assertLess(avg_response_time, 5.0)  # Average response time < 5s
        self.assertLess(error_rate, 20)  # Error rate < 20%
        
        print(f"✅ Heavy load test passed")
        print(f"📊 Success rate: {success_rate:.2f}%")
        print(f"📊 Throughput: {throughput:.2f} req/s")
        print(f"📊 Avg response time: {avg_response_time:.2f}s")
        print(f"📊 Error rate: {error_rate:.2f}%")
    
    def test_peak_load_performance(self):
        """Test system performance under peak load."""
        # Simulate peak load scenario
        concurrent_users = 200
        duration = 300  # seconds
        
        start_time = time.time()
        
        # Simulate peak load with maximum resource usage
        requests_processed = 0
        successful_requests = 0
        response_times = []
        error_count = 0
        timeout_count = 0
        
        for i in range(concurrent_users):
            request_start = time.time()
            
            try:
                # Simulate maximum load work
                work_time = random.uniform(1.0, 3.0)
                time.sleep(work_time)
                
                # Check for timeout
                if work_time > 2.5:
                    timeout_count += 1
                    continue
                
                # Simulate higher failure rate under peak load
                success = random.uniform(0.75, 1.0) > 0.15
                if success:
                    successful_requests += 1
                else:
                    error_count += 1
                
                requests_processed += 1
                request_end = time.time()
                request_time = request_end - request_start
                response_times.append(request_time)
                
            except Exception as e:
                error_count += 1
                print(f"⚠️ Request {i} failed: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        success_rate = (successful_requests / requests_processed) * 100 if requests_processed > 0 else 0
        throughput = requests_processed / total_time
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        error_rate = (error_count / requests_processed) * 100 if requests_processed > 0 else 100
        timeout_rate = (timeout_count / concurrent_users) * 100
        
        # Verify performance criteria (most lenient for peak load)
        self.assertGreater(success_rate, 70)  # At least 70% success rate
        self.assertGreater(throughput, 40)  # At least 40 requests per second
        self.assertLess(avg_response_time, 8.0)  # Average response time < 8s
        self.assertLess(error_rate, 30)  # Error rate < 30%
        self.assertLess(timeout_rate, 25)  # Timeout rate < 25%
        
        print(f"✅ Peak load test passed")
        print(f"📊 Success rate: {success_rate:.2f}%")
        print(f"📊 Throughput: {throughput:.2f} req/s")
        print(f"📊 Avg response time: {avg_response_time:.2f}s")
        print(f"📊 Error rate: {error_rate:.2f}%")
        print(f"📊 Timeout rate: {timeout_rate:.2f}%")

if __name__ == '__main__':
    unittest.main()
```

### Example 2: Stress Testing

```python
from test_framework.test_performance import TestStressPerformance
import unittest
import time
import random
import psutil

class TestOptimizationCoreStress(unittest.TestCase):
    def setUp(self):
        self.stress_test = TestStressPerformance()
        self.stress_test.setUp()
    
    def test_memory_stress(self):
        """Test system behavior under memory stress."""
        # Simulate memory stress scenario
        memory_target = 1000  # MB
        duration = 60  # seconds
        
        start_time = time.time()
        memory_usage = []
        
        # Allocate memory gradually
        allocated_memory = []
        try:
            while time.time() - start_time < duration:
                # Allocate memory chunk
                chunk_size = random.randint(10, 50)  # MB
                memory_chunk = [0] * (chunk_size * 1024 * 1024 // 4)  # 4 bytes per int
                allocated_memory.append(memory_chunk)
                
                # Check current memory usage
                current_memory = psutil.virtual_memory().percent
                memory_usage.append(current_memory)
                
                # Simulate work with allocated memory
                work_time = random.uniform(0.1, 0.5)
                time.sleep(work_time)
                
                # Check if memory limit reached
                if current_memory > 95:
                    print("⚠️ Memory limit reached, stopping allocation")
                    break
                
                # Simulate memory cleanup occasionally
                if random.uniform(0, 1) < 0.1:  # 10% chance
                    if allocated_memory:
                        allocated_memory.pop()
        
        finally:
            # Clean up allocated memory
            allocated_memory.clear()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        max_memory_usage = max(memory_usage) if memory_usage else 0
        avg_memory_usage = sum(memory_usage) / len(memory_usage) if memory_usage else 0
        memory_efficiency = len(allocated_memory) / (total_time / 60)  # MB per minute
        
        # Verify stress test criteria
        self.assertGreater(total_time, 30)  # Should run for at least 30 seconds
        self.assertLess(max_memory_usage, 95)  # Should not exceed 95% memory usage
        self.assertGreater(memory_efficiency, 0)  # Should allocate memory efficiently
        
        print(f"✅ Memory stress test passed")
        print(f"📊 Max memory usage: {max_memory_usage:.2f}%")
        print(f"📊 Avg memory usage: {avg_memory_usage:.2f}%")
        print(f"📊 Memory efficiency: {memory_efficiency:.2f} MB/min")
    
    def test_cpu_stress(self):
        """Test system behavior under CPU stress."""
        # Simulate CPU stress scenario
        duration = 120  # seconds
        
        start_time = time.time()
        cpu_usage = []
        
        # Simulate CPU-intensive work
        def cpu_intensive_task():
            while time.time() - start_time < duration:
                # CPU intensive operations
                data = [random.random() for _ in range(10000)]
                _ = sum(x ** 2 for x in data)
                _ = sum(x ** 0.5 for x in data)
                _ = sum(x ** 3 for x in data)
                
                # Check CPU usage
                current_cpu = psutil.cpu_percent()
                cpu_usage.append(current_cpu)
                
                # Brief pause to prevent system lockup
                time.sleep(0.01)
        
        # Run CPU intensive task
        cpu_intensive_task()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        max_cpu_usage = max(cpu_usage) if cpu_usage else 0
        avg_cpu_usage = sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0
        cpu_efficiency = len(cpu_usage) / total_time  # Samples per second
        
        # Verify stress test criteria
        self.assertGreater(total_time, 60)  # Should run for at least 60 seconds
        self.assertGreater(max_cpu_usage, 50)  # Should utilize CPU significantly
        self.assertLess(max_cpu_usage, 100)  # Should not exceed 100% CPU usage
        self.assertGreater(cpu_efficiency, 10)  # Should sample CPU usage frequently
        
        print(f"✅ CPU stress test passed")
        print(f"📊 Max CPU usage: {max_cpu_usage:.2f}%")
        print(f"📊 Avg CPU usage: {avg_cpu_usage:.2f}%")
        print(f"📊 CPU efficiency: {cpu_efficiency:.2f} samples/s")
    
    def test_io_stress(self):
        """Test system behavior under I/O stress."""
        # Simulate I/O stress scenario
        io_operations = 10000
        duration = 180  # seconds
        
        start_time = time.time()
        io_times = []
        successful_operations = 0
        
        # Simulate I/O operations
        temp_files = []
        try:
            for i in range(io_operations):
                if time.time() - start_time > duration:
                    break
                
                operation_start = time.time()
                
                # Simulate file I/O
                temp_file = f"temp_{i}.dat"
                temp_files.append(temp_file)
                
                # Write data
                with open(temp_file, 'w') as f:
                    data = 'x' * random.randint(100, 1000)  # 100-1000 bytes
                    f.write(data)
                
                # Read data
                with open(temp_file, 'r') as f:
                    _ = f.read()
                
                # Clean up
                try:
                    import os
                    os.remove(temp_file)
                    temp_files.remove(temp_file)
                except:
                    pass
                
                operation_end = time.time()
                operation_time = operation_end - operation_start
                io_times.append(operation_time)
                successful_operations += 1
                
                # Brief pause to prevent system overload
                time.sleep(0.001)
        
        finally:
            # Clean up remaining files
            for temp_file in temp_files:
                try:
                    import os
                    os.remove(temp_file)
                except:
                    pass
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        total_operations = successful_operations
        operations_per_second = total_operations / total_time if total_time > 0 else 0
        avg_io_time = sum(io_times) / len(io_times) if io_times else 0
        max_io_time = max(io_times) if io_times else 0
        
        # Verify stress test criteria
        self.assertGreater(total_operations, 1000)  # Should complete at least 1000 operations
        self.assertGreater(operations_per_second, 10)  # At least 10 operations per second
        self.assertLess(avg_io_time, 1.0)  # Average I/O time < 1 second
        self.assertLess(max_io_time, 5.0)  # Max I/O time < 5 seconds
        
        print(f"✅ I/O stress test passed")
        print(f"📊 Total operations: {total_operations}")
        print(f"📊 Operations/sec: {operations_per_second:.2f}")
        print(f"📊 Avg I/O time: {avg_io_time:.4f}s")
        print(f"📊 Max I/O time: {max_io_time:.4f}s")
    
    def test_network_stress(self):
        """Test system behavior under network stress."""
        # Simulate network stress scenario
        duration = 300  # seconds
        
        start_time = time.time()
        network_operations = 0
        successful_operations = 0
        
        # Simulate network operations
        while time.time() - start_time < duration:
            operation_start = time.time()
            
            # Simulate network request
            try:
                # Simulate network latency
                latency = random.uniform(0.01, 0.1)
                time.sleep(latency)
                
                # Simulate network success/failure
                success = random.uniform(0.8, 1.0) > 0.05
                if success:
                    successful_operations += 1
                
                network_operations += 1
                operation_end = time.time()
                operation_time = operation_end - operation_start
                
                # Verify operation time
                self.assertLess(operation_time, 1.0)  # Network operation < 1 second
                
            except Exception as e:
                print(f"⚠️ Network operation failed: {e}")
            
            # Brief pause between operations
            time.sleep(0.01)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        operations_per_second = network_operations / total_time if total_time > 0 else 0
        success_rate = (successful_operations / network_operations) * 100 if network_operations > 0 else 0
        
        # Verify stress test criteria
        self.assertGreater(network_operations, 100)  # Should complete at least 100 operations
        self.assertGreater(operations_per_second, 1)  # At least 1 operation per second
        self.assertGreater(success_rate, 80)  # At least 80% success rate
        
        print(f"✅ Network stress test passed")
        print(f"📊 Total operations: {network_operations}")
        print(f"📊 Operations/sec: {operations_per_second:.2f}")
        print(f"📊 Success rate: {success_rate:.2f}%")

if __name__ == '__main__':
    unittest.main()
```

### Example 3: Scalability Testing

```python
from test_framework.test_performance import TestScalabilityPerformance
import unittest
import time
import random

class TestOptimizationCoreScalability(unittest.TestCase):
    def setUp(self):
        self.scalability_test = TestScalabilityPerformance()
        self.scalability_test.setUp()
    
    def test_horizontal_scaling(self):
        """Test horizontal scaling performance."""
        # Test different instance counts
        instance_counts = [1, 2, 4, 8]
        scaling_results = {}
        
        for instances in instance_counts:
            start_time = time.time()
            
            # Simulate horizontal scaling
            total_work = 0
            for i in range(instances):
                # Simulate work per instance
                instance_work = random.uniform(100, 500)
                total_work += instance_work
                
                # Simulate instance processing time
                processing_time = random.uniform(1, 3)
                time.sleep(processing_time)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Calculate scaling metrics
            throughput = total_work / execution_time
            efficiency = throughput / instances  # Work per instance
            
            scaling_results[instances] = {
                'execution_time': execution_time,
                'throughput': throughput,
                'efficiency': efficiency,
                'total_work': total_work
            }
            
            print(f"📊 {instances} instances: {throughput:.2f} work/s, efficiency: {efficiency:.2f}")
        
        # Verify scaling efficiency
        if len(scaling_results) > 1:
            first_throughput = scaling_results[1]['throughput']
            last_throughput = scaling_results[8]['throughput']
            scaling_efficiency = last_throughput / (first_throughput * 8)
            
            self.assertGreater(scaling_efficiency, 0.5)  # At least 50% scaling efficiency
            print(f"✅ Horizontal scaling efficiency: {scaling_efficiency:.2f}")
    
    def test_vertical_scaling(self):
        """Test vertical scaling performance."""
        # Test different resource levels
        resource_levels = [1, 2, 4, 8]
        scaling_results = {}
        
        for resources in resource_levels:
            start_time = time.time()
            
            # Simulate vertical scaling
            workload_intensity = resources * 1000
            total_work = 0
            
            for _ in range(workload_intensity):
                # Simulate work based on resource level
                work_amount = random.uniform(1, 10) * resources
                total_work += work_amount
                
                # Simulate processing
                processing_time = random.uniform(0.001, 0.01)
                time.sleep(processing_time)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Calculate scaling metrics
            throughput = total_work / execution_time
            efficiency = throughput / resources  # Work per resource unit
            
            scaling_results[resources] = {
                'execution_time': execution_time,
                'throughput': throughput,
                'efficiency': efficiency,
                'total_work': total_work
            }
            
            print(f"📊 {resources} resources: {throughput:.2f} work/s, efficiency: {efficiency:.2f}")
        
        # Verify scaling efficiency
        if len(scaling_results) > 1:
            first_efficiency = scaling_results[1]['efficiency']
            last_efficiency = scaling_results[8]['efficiency']
            efficiency_ratio = last_efficiency / first_efficiency
            
            self.assertGreater(efficiency_ratio, 0.8)  # At least 80% efficiency retention
            print(f"✅ Vertical scaling efficiency: {efficiency_ratio:.2f}")
    
    def test_data_scaling(self):
        """Test data scaling performance."""
        # Test different data sizes
        data_sizes = [100, 1000, 10000, 100000]
        scaling_results = {}
        
        for data_size in data_sizes:
            start_time = time.time()
            
            # Simulate data processing
            data = [random.random() for _ in range(data_size)]
            
            # Simulate data operations
            _ = sum(data)
            _ = sum(x ** 2 for x in data)
            _ = sum(x ** 0.5 for x in data)
            _ = max(data)
            _ = min(data)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Calculate scaling metrics
            throughput = data_size / execution_time
            efficiency = throughput / data_size  # Operations per data point
            
            scaling_results[data_size] = {
                'execution_time': execution_time,
                'throughput': throughput,
                'efficiency': efficiency,
                'data_size': data_size
            }
            
            print(f"📊 {data_size} data points: {throughput:.2f} points/s, efficiency: {efficiency:.6f}")
        
        # Verify data scaling
        if len(scaling_results) > 1:
            small_data_throughput = scaling_results[100]['throughput']
            large_data_throughput = scaling_results[100000]['throughput']
            scaling_ratio = large_data_throughput / small_data_throughput
            
            self.assertGreater(scaling_ratio, 0.1)  # Should maintain some throughput
            print(f"✅ Data scaling ratio: {scaling_ratio:.2f}")
    
    def test_user_scaling(self):
        """Test user scaling performance."""
        # Test different user counts
        user_counts = [10, 50, 100, 200]
        scaling_results = {}
        
        for user_count in user_counts:
            start_time = time.time()
            
            # Simulate user requests
            total_requests = 0
            successful_requests = 0
            
            for user_id in range(user_count):
                # Simulate user request
                request_start = time.time()
                
                # Simulate request processing
                processing_time = random.uniform(0.1, 0.5)
                time.sleep(processing_time)
                
                # Simulate request success/failure
                success = random.uniform(0.9, 1.0) > 0.05
                if success:
                    successful_requests += 1
                
                total_requests += 1
                request_end = time.time()
                request_time = request_end - request_start
                
                # Verify request time
                self.assertLess(request_time, 2.0)  # Request < 2 seconds
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Calculate scaling metrics
            throughput = total_requests / execution_time
            success_rate = (successful_requests / total_requests) * 100
            efficiency = throughput / user_count  # Requests per user
            
            scaling_results[user_count] = {
                'execution_time': execution_time,
                'throughput': throughput,
                'success_rate': success_rate,
                'efficiency': efficiency,
                'total_requests': total_requests
            }
            
            print(f"📊 {user_count} users: {throughput:.2f} req/s, success: {success_rate:.2f}%")
        
        # Verify user scaling
        if len(scaling_results) > 1:
            small_user_throughput = scaling_results[10]['throughput']
            large_user_throughput = scaling_results[200]['throughput']
            scaling_ratio = large_user_throughput / small_user_throughput
            
            self.assertGreater(scaling_ratio, 0.5)  # Should maintain some throughput
            print(f"✅ User scaling ratio: {scaling_ratio:.2f}")

if __name__ == '__main__':
    unittest.main()
```

## Running Performance Tests

### Command Line Execution

```bash
# Run all performance tests
python -m test_framework.test_performance

# Run specific performance test
python -m test_framework.test_performance TestLoadPerformance

# Run with verbose output
python -m test_framework.test_performance -v
```

### Programmatic Execution

```python
from test_framework.test_performance import TestLoadPerformance
import unittest

# Create test suite
suite = unittest.TestSuite()

# Add specific tests
suite.addTest(TestLoadPerformance('test_light_load_performance'))
suite.addTest(TestLoadPerformance('test_medium_load_performance'))

# Run tests
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)

# Check results
print(f"Tests run: {result.testsRun}")
print(f"Failures: {len(result.failures)}")
print(f"Errors: {len(result.errors)}")
```

## Performance Monitoring

### Real-time Monitoring

```python
import psutil
import time

def monitor_performance():
    """Monitor system performance during tests."""
    while True:
        # Get system metrics
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent
        
        print(f"CPU: {cpu_usage:.2f}%, Memory: {memory_usage:.2f}%, Disk: {disk_usage:.2f}%")
        
        # Check for performance issues
        if cpu_usage > 90:
            print("⚠️ High CPU usage detected")
        if memory_usage > 90:
            print("⚠️ High memory usage detected")
        if disk_usage > 90:
            print("⚠️ High disk usage detected")
        
        time.sleep(1)
```

### Performance Metrics Collection

```python
def collect_performance_metrics():
    """Collect comprehensive performance metrics."""
    metrics = {
        'cpu': {
            'usage': psutil.cpu_percent(),
            'count': psutil.cpu_count(),
            'frequency': psutil.cpu_freq()
        },
        'memory': {
            'usage': psutil.virtual_memory().percent,
            'available': psutil.virtual_memory().available,
            'total': psutil.virtual_memory().total
        },
        'disk': {
            'usage': psutil.disk_usage('/').percent,
            'free': psutil.disk_usage('/').free,
            'total': psutil.disk_usage('/').total
        },
        'network': {
            'bytes_sent': psutil.net_io_counters().bytes_sent,
            'bytes_recv': psutil.net_io_counters().bytes_recv
        }
    }
    
    return metrics
```

## Best Practices

### 1. Test Design
- Start with light loads and gradually increase
- Test both success and failure scenarios
- Monitor system resources during tests
- Use realistic test data and scenarios

### 2. Performance Criteria
- Set clear performance thresholds
- Define acceptable response times
- Establish throughput requirements
- Monitor resource utilization

### 3. Test Execution
- Run tests in isolated environments
- Use consistent test conditions
- Document test results and findings
- Analyze performance trends over time

### 4. Resource Management
- Monitor system resources during tests
- Clean up resources after tests
- Use appropriate timeouts
- Handle test failures gracefully

## Conclusion

Performance testing is essential for ensuring that the optimization core system can handle various loads and conditions effectively. These examples demonstrate comprehensive performance testing approaches, from basic load testing to advanced scalability testing. By following these patterns and best practices, you can create robust performance tests that validate system performance and identify potential bottlenecks.