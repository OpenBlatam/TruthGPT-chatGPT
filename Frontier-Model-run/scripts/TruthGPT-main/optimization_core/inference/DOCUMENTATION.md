# 📚 Inference Module Documentation

Comprehensive documentation for the TruthGPT inference module.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Performance Guide](#performance-guide)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements_advanced.txt

# Or minimal install
pip install fastapi uvicorn httpx pydantic redis
```

### Basic Usage

```python
from inference import create_inference_engine, EngineType

# Create engine
engine = create_inference_engine(
    model="gpt-4o",
    engine_type=EngineType.AUTO
)

# Generate text
result = engine.generate(
    prompts="Hello, world!",
    max_tokens=128,
    temperature=0.7
)

print(result)
```

### API Server

```bash
# Start API server
python -m uvicorn inference.api:app --host 0.0.0.0 --port 8080

# Or using Docker
docker-compose up -d
```

## 🏗️ Architecture

### Engine Types

| Engine | Performance | GPU Required | Best For |
|--------|-----------|--------------|----------|
| **vLLM** | 5-10x faster | Yes | General purpose, high throughput |
| **TensorRT-LLM** | 10-20x faster | NVIDIA GPU | Maximum performance on NVIDIA |
| **Async vLLM** | 5-10x faster | Yes | High-throughput serving |

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Inference API Server                     │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   FastAPI    │  │  Middleware  │  │  Monitoring  │    │
│  │   Endpoints  │  │  Stack       │  │  & Metrics  │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                  │                  │             │
│  ┌──────▼──────────────────▼──────────────────▼─────────┐│
│  │              Engine Factory                             ││
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            ││
│  │  │  vLLM    │  │TensorRT- │  │  Base    │            ││
│  │  │ Engine   │  │  LLM     │  │ Engine   │            ││
│  │  └──────────┘  └──────────┘  └──────────┘            ││
│  └───────────────────────────────────────────────────────┘│
│                                                             │
│  ┌───────────────────────────────────────────────────────┐│
│  │              Supporting Services                        ││
│  │  Cache │ Rate Limiter │ Circuit Breaker │ Metrics     ││
│  └───────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## 📖 API Reference

### Core Classes

#### `BaseInferenceEngine`

Abstract base class for all inference engines.

```python
class BaseInferenceEngine(ABC):
    @abstractmethod
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs
    ) -> Union[str, List[str]]:
        """Generate text from prompts."""
```

#### `VLLMEngine`

High-performance engine using vLLM.

```python
engine = VLLMEngine(
    model="gpt-4o",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    max_model_len=4096,
    dtype="auto",
    quantization=None
)
```

#### `TensorRTLLMEngine`

Maximum performance engine using TensorRT-LLM.

```python
engine = TensorRTLLMEngine(
    model_path="/path/to/model",
    engine_dir="/path/to/engine",
    max_batch_size=32,
    max_sequence_length=4096
)
```

### Factory Functions

#### `create_inference_engine()`

Automatically select and create the best available engine.

```python
engine = create_inference_engine(
    model="gpt-4o",
    engine_type=EngineType.AUTO,  # or VLLM, TENSORRT_LLM, ASYNC_VLLM
    prefer_gpu=True,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9
)
```

#### `list_available_engines()`

Check which engines are available.

```python
engines = list_available_engines()
# Returns: {'vllm': True, 'tensorrt_llm': False, 'async_vllm': True}
```

## ⚡ Performance Guide

### Optimization Strategies

#### 1. Batching

```python
# Increase batch size for better GPU utilization
BATCH_MAX_SIZE=64
BATCH_FLUSH_TIMEOUT_MS=10
```

**Expected Impact:**
- 20-40% latency reduction
- 2-3x throughput increase

#### 2. Caching

```python
# Enable Redis caching
CACHE_BACKEND=redis
REDIS_URL=redis://localhost:6379/0
```

**Expected Impact:**
- 50-80% latency reduction for cached requests
- Reduced GPU load

#### 3. Engine Selection

```python
# Use TensorRT-LLM for maximum performance
engine = create_inference_engine(
    model="gpt-4o",
    engine_type=EngineType.TENSORRT_LLM
)
```

**Expected Impact:**
- 10-20x faster than PyTorch
- Lower memory usage

### Performance Metrics

| Metric | Target | How to Monitor |
|--------|--------|----------------|
| **p95 Latency** | <300ms | Prometheus `/metrics` |
| **p99 Latency** | <500ms | Prometheus `/metrics` |
| **Throughput** | >100 RPS | Prometheus `/metrics` |
| **Cache Hit Rate** | >50% | Prometheus `/metrics` |
| **Error Rate** | <1% | Prometheus `/metrics` |

## 🚢 Deployment

### Docker

```bash
# Build image
docker build -t frontier-inference-api -f inference/Dockerfile .

# Run container
docker run -p 8080:8080 \
  -e TRUTHGPT_API_TOKEN=your-token \
  -e TRUTHGPT_CONFIG=configs/llm_default.yaml \
  frontier-inference-api
```

### Docker Compose

```bash
# Start full stack
docker-compose up -d

# Services:
# - API: http://localhost:8080
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000
# - Redis: localhost:6379
```

### Kubernetes

```bash
# Apply deployment
kubectl apply -f k8s/deployment.yaml

# Check status
kubectl get pods -n inference
kubectl get svc -n inference
```

## 📊 Monitoring

### Prometheus Metrics

Available at `/metrics` endpoint:

- `inference_requests_total` - Total requests
- `inference_request_duration_ms` - Request latency histogram
- `inference_errors_5xx_total` - Server errors
- `inference_cache_hits_total` - Cache hits
- `inference_queue_depth` - Queue depth
- `circuit_breaker_open_total` - Circuit breaker events

### Grafana Dashboard

Import dashboard from `grafana/dashboards/inference-api.json` or use auto-provisioned dashboard with docker-compose.

### Logging

Structured JSON logs with:
- Request ID correlation
- Model and endpoint information
- Latency metrics
- Error details

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRUTHGPT_API_TOKEN` | `changeme` | API authentication token |
| `TRUTHGPT_CONFIG` | `configs/llm_default.yaml` | Model configuration |
| `PORT` | `8080` | API server port |
| `BATCH_MAX_SIZE` | `32` | Maximum batch size |
| `BATCH_FLUSH_TIMEOUT_MS` | `20` | Batch flush timeout (ms) |
| `RATE_LIMIT_RPM` | `600` | Requests per minute |
| `CACHE_BACKEND` | `memory` | Cache backend (memory/redis) |
| `REDIS_URL` | - | Redis connection URL |
| `ENABLE_METRICS` | `true` | Enable Prometheus metrics |
| `ENABLE_TRACING` | `true` | Enable OpenTelemetry tracing |

## 🐛 Troubleshooting

### Common Issues

#### 1. Model Not Loading

**Symptom:** `ERROR: Model not found`

**Solutions:**
- Check `TRUTHGPT_CONFIG` path
- Verify model files exist
- Check model format (HuggingFace compatible)

#### 2. Redis Connection Failed

**Symptom:** `ERROR: Could not connect to Redis`

**Solutions:**
- Verify `REDIS_URL` is correct
- Check Redis is running: `redis-cli ping`
- Check network connectivity

#### 3. High Latency

**Symptom:** p95 latency > 500ms

**Solutions:**
- Increase batch size
- Enable caching
- Use TensorRT-LLM engine
- Check GPU utilization

#### 4. Rate Limit Errors

**Symptom:** `429 Too Many Requests`

**Solutions:**
- Adjust `RATE_LIMIT_RPM`
- Implement client-side rate limiting
- Use request queuing

### Debug Mode

```bash
export ENVIRONMENT=development
export LOG_LEVEL=DEBUG
python -m uvicorn inference.api:app --log-level debug
```

## 📚 Additional Resources

- [Quick Start Guide](./QUICK_START.md) - 5-minute setup guide
- [Performance Guide](./PERFORMANCE_GUIDE.md) - Detailed optimization guide
- [API Reference](./README.md) - Complete API documentation

---

**Version:** 2.0.0  
**Status:** ✅ Production Ready









