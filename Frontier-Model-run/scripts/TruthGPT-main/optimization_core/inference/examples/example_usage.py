"""
📚 Usage Examples
Complete examples for using the inference API
"""

import asyncio
import httpx
import json
from typing import List


# ============================================================================
# Example 1: Basic Synchronous Inference
# ============================================================================

def example_sync_inference():
    """Basic synchronous inference example"""
    import requests
    
    response = requests.post(
        "http://localhost:8080/v1/infer",
        headers={
            "Authorization": "Bearer your-token",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-4o",
            "prompt": "Explain quantum computing in simple terms",
            "params": {
                "max_new_tokens": 128,
                "temperature": 0.7,
                "top_p": 0.9
            },
            "idempotency_key": "unique-request-id-123"
        },
        timeout=30.0
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"Output: {data['output']}")
        print(f"Latency: {data['latency_ms']}ms")
        print(f"Cached: {data['cached']}")
        print(f"Usage: {data['usage']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")


# ============================================================================
# Example 2: Streaming Inference
# ============================================================================

def example_streaming_inference():
    """Streaming inference example"""
    import requests
    
    response = requests.post(
        "http://localhost:8080/v1/infer/stream",
        headers={
            "Authorization": "Bearer your-token",
            "Accept": "text/event-stream",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-4o",
            "prompt": "Write a short story",
            "params": {
                "max_new_tokens": 256,
                "temperature": 0.8
            }
        },
        stream=True
    )
    
    response.raise_for_status()
    
    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith('data:'):
                data_str = line_str[5:].strip()
                if data_str:
                    try:
                        data = json.loads(data_str)
                        if 'text' in data:
                            print(data['text'], end='', flush=True)
                        if data.get('finish_reason'):
                            print("\n[Done]")
                            break
                    except json.JSONDecodeError:
                        pass


# ============================================================================
# Example 3: Async Batch Processing
# ============================================================================

async def example_async_batch(prompts: List[str]):
    """Process multiple prompts asynchronously"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = []
        
        for prompt in prompts:
            task = client.post(
                "http://localhost:8080/v1/infer",
                headers={"Authorization": "Bearer your-token"},
                json={
                    "model": "gpt-4o",
                    "prompt": prompt,
                    "params": {"max_new_tokens": 64}
                }
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, httpx.Response) and response.status_code == 200:
                data = response.json()
                results.append({
                    "prompt": prompts[i],
                    "output": data["output"],
                    "latency_ms": data["latency_ms"]
                })
            else:
                results.append({
                    "prompt": prompts[i],
                    "error": str(response)
                })
        
        return results


# ============================================================================
# Example 4: Health Monitoring
# ============================================================================

def example_health_check():
    """Monitor API health"""
    import requests
    import time
    
    while True:
        try:
            response = requests.get("http://localhost:8080/health", timeout=5)
            data = response.json()
            
            status = data.get("status", "unknown")
            checks = data.get("checks", {})
            
            print(f"[{time.strftime('%H:%M:%S')}] Status: {status}")
            for component, status in checks.items():
                print(f"  {component}: {status}")
            
            if status == "healthy":
                print("✓ API is healthy")
            else:
                print("⚠ API is degraded")
        
        except Exception as e:
            print(f"✗ Error: {e}")
        
        time.sleep(10)


# ============================================================================
# Example 5: Metrics Collection
# ============================================================================

def example_collect_metrics():
    """Collect and parse Prometheus metrics"""
    import requests
    import re
    
    response = requests.get("http://localhost:8080/metrics")
    metrics_text = response.text
    
    # Parse metrics
    metrics = {}
    for line in metrics_text.split("\n"):
        if line and not line.startswith("#"):
            parts = line.split()
            if len(parts) >= 2:
                metric_name = parts[0].split("{")[0]
                try:
                    metric_value = float(parts[1])
                    metrics[metric_name] = metric_value
                except ValueError:
                    pass
    
    # Display key metrics
    print("Key Metrics:")
    print(f"  Requests: {metrics.get('inference_requests_total', 0)}")
    print(f"  Errors (5xx): {metrics.get('inference_errors_5xx_total', 0)}")
    print(f"  Cache Hits: {metrics.get('inference_cache_hits_total', 0)}")
    print(f"  Avg Latency: {metrics.get('inference_request_duration_ms', 0)}ms")
    print(f"  Queue Depth: {metrics.get('inference_queue_depth', 0)}")
    
    return metrics


# ============================================================================
# Example 6: Webhook Integration
# ============================================================================

def example_webhook_send(payload: dict):
    """Send webhook with HMAC signature"""
    import hmac
    import hashlib
    import requests
    import time
    
    secret = "your-webhook-secret"
    timestamp = int(time.time())
    
    # Create signature
    message = f"{timestamp}.{json.dumps(payload, sort_keys=True)}"
    signature = hmac.new(
        secret.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    
    # Send webhook
    response = requests.post(
        "http://localhost:8080/webhooks/ingest",
        headers={
            "X-Signature": signature,
            "X-Timestamp": str(timestamp),
            "Idempotency-Key": f"webhook-{timestamp}",
            "Content-Type": "application/json"
        },
        json=payload
    )
    
    return response.json()


# ============================================================================
# Example 7: Performance Testing
# ============================================================================

async def example_performance_test(num_requests: int = 100, concurrency: int = 10):
    """Run performance test"""
    import time
    from statistics import mean, median
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        latencies = []
        successes = 0
        errors = 0
        
        start_time = time.time()
        
        semaphore = asyncio.Semaphore(concurrency)
        
        async def make_request():
            nonlocal successes, errors
            async with semaphore:
                req_start = time.time()
                try:
                    response = await client.post(
                        "http://localhost:8080/v1/infer",
                        headers={"Authorization": "Bearer your-token"},
                        json={
                            "model": "gpt-4o",
                            "prompt": "Test prompt",
                            "params": {"max_new_tokens": 32}
                        }
                    )
                    req_latency = (time.time() - req_start) * 1000
                    
                    if response.status_code == 200:
                        successes += 1
                        latencies.append(req_latency)
                    else:
                        errors += 1
                except Exception:
                    errors += 1
        
        tasks = [make_request() for _ in range(num_requests)]
        await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        print(f"\nPerformance Test Results:")
        print(f"  Total Requests: {num_requests}")
        print(f"  Successful: {successes}")
        print(f"  Errors: {errors}")
        print(f"  Duration: {total_time:.2f}s")
        print(f"  Throughput: {successes/total_time:.2f} req/s")
        
        if latencies:
            print(f"  Avg Latency: {mean(latencies):.2f}ms")
            print(f"  Median Latency: {median(latencies):.2f}ms")
            print(f"  Min Latency: {min(latencies):.2f}ms")
            print(f"  Max Latency: {max(latencies):.2f}ms")


# ============================================================================
# Example 8: Error Handling and Retries
# ============================================================================

def example_with_retries(prompt: str, max_retries: int = 3):
    """Inference with automatic retries"""
    import requests
    import time
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:8080/v1/infer",
                headers={"Authorization": "Bearer your-token"},
                json={
                    "model": "gpt-4o",
                    "prompt": prompt,
                    "params": {}
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                return response.json()
            
            # Rate limited
            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 5))
                print(f"Rate limited. Retrying after {retry_after}s...")
                time.sleep(retry_after)
                continue
            
            # Server error - retry
            elif response.status_code >= 500:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Server error. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            # Client error - don't retry
            else:
                raise Exception(f"Client error: {response.status_code}")
        
        except requests.exceptions.Timeout:
            wait_time = 2 ** attempt
            print(f"Timeout. Retrying in {wait_time}s...")
            time.sleep(wait_time)
        
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt
            print(f"Error: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
    
    raise Exception("Max retries exceeded")


# ============================================================================
# Main Examples
# ============================================================================

if __name__ == "__main__":
    print("Running examples...")
    
    # Example 1: Basic inference
    print("\n1. Basic Inference:")
    try:
        example_sync_inference()
    except Exception as e:
        print(f"  Error: {e}")
    
    # Example 4: Health check
    print("\n2. Health Check:")
    try:
        example_health_check()
    except KeyboardInterrupt:
        print("\n  Stopped")
    
    # Example 5: Metrics
    print("\n3. Metrics Collection:")
    try:
        example_collect_metrics()
    except Exception as e:
        print(f"  Error: {e}")
    
    # Example 7: Performance test
    print("\n4. Performance Test:")
    try:
        asyncio.run(example_performance_test(num_requests=10, concurrency=5))
    except Exception as e:
        print(f"  Error: {e}")



