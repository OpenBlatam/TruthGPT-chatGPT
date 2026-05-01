# 🚀 Frontier Model Run - Inference API

Enterprise-grade inference API with advanced features for production deployment.

## 🎯 Features

- ✅ **Dynamic Batching** - Efficient request batching with configurable timeouts
- ✅ **Streaming Support** - Server-Sent Events (SSE) for real-time responses
- ✅ **Rate Limiting** - Sliding window rate limiting per client/endpoint
- ✅ **Circuit Breakers** - Resilient failure handling with automatic recovery
- ✅ **Distributed Caching** - Redis-backed caching with LRU eviction
- ✅ **Prometheus Metrics** - Comprehensive metrics with histograms and percentiles
- ✅ **OpenTelemetry Tracing** - Distributed tracing for observability
- ✅ **Structured Logging** - JSON logging with request context
- ✅ **Health Checks** - Kubernetes-ready health/readiness probes
- ✅ **Webhook Support** - HMAC-signed webhook ingestion

## 📦 Quick Start

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f inference-api

# Access services
# API: http://localhost:8080
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
# Redis: localhost:6379
```

### Using Kubernetes

```bash
# Create namespace
kubectl create namespace inference

# Apply configurations
kubectl apply -f k8s/deployment.yaml

# Check status
kubectl get pods -n inference
kubectl get svc -n inference
```

### Local Development

```bash
# Install dependencies
pip install -r requirements_advanced.txt

# Set environment variables
export TRUTHGPT_API_TOKEN=your-token
export TRUTHGPT_CONFIG=configs/llm_default.yaml

# Run API
python -m uvicorn inference.api:app --host 0.0.0.0 --port 8080 --reload
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRUTHGPT_API_TOKEN` | `changeme` | API authentication token |
| `TRUTHGPT_CONFIG` | `configs/llm_default.yaml` | Model configuration path |
| `PORT` | `8080` | API server port |
| `BATCH_MAX_SIZE` | `32` | Maximum batch size |
| `BATCH_FLUSH_TIMEOUT_MS` | `20` | Batch flush timeout (ms) |
| `RATE_LIMIT_RPM` | `600` | Requests per minute limit |
| `CACHE_BACKEND` | `memory` | Cache backend (memory/redis) |
| `REDIS_URL` | - | Redis connection URL |
| `ENABLE_METRICS` | `true` | Enable Prometheus metrics |
| `ENABLE_TRACING` | `true` | Enable OpenTelemetry tracing |

See [INFERENCE_API_IMPROVEMENTS.md](../../INFERENCE_API_IMPROVEMENTS.md) for full documentation.

## 📊 API Endpoints

### Health & Status

- `GET /` - API information
- `GET /health` - Health check
- `GET /ready` - Readiness check
- `GET /metrics` - Prometheus metrics

### Inference

- `POST /v1/infer` - Synchronous inference
- `POST /v1/infer/stream` - Streaming inference (SSE)

### Webhooks

- `POST /webhooks/ingest` - Webhook ingestion

## 📈 Monitoring

### Prometheus Metrics

Available at `/metrics` endpoint:

- `inference_requests_total` - Total requests
- `inference_request_duration_ms` - Request latency
- `inference_errors_5xx_total` - Server errors
- `inference_cache_hits_total` - Cache hits
- `inference_queue_depth` - Queue depth
- `circuit_breaker_open_total` - Open circuits

### Grafana Dashboard

Import the dashboard from `grafana/dashboards/inference-api.json` or use the auto-provisioned dashboard when running with docker-compose.

### Logging

Structured JSON logs with:
- Request ID correlation
- Model and endpoint information
- Latency metrics
- Error details

## 🔒 Security

- Bearer token authentication
- Rate limiting per IP/client
- HMAC webhook signature verification
- Input validation and sanitization
- Circuit breakers to prevent cascading failures

## 🚀 Deployment

### Docker

```bash
docker build -t frontier-inference-api -f inference/Dockerfile .
docker run -p 8080:8080 frontier-inference-api
```

### Kubernetes

See `k8s/deployment.yaml` for complete Kubernetes manifests including:
- Deployment with 3 replicas
- Service with ClusterIP
- HorizontalPodAutoscaler (2-10 replicas)
- ConfigMaps and Secrets
- Health checks (liveness/readiness)

### CI/CD

GitHub Actions workflow (`.github/workflows/ci-cd.yml`) includes:
- Automated testing
- Security scanning
- Docker image building
- Multi-stage deployment
- Load testing

## 🧪 Testing

```bash
# Unit tests
pytest tests/ -v

# Integration tests
pytest tests/integration/ -v

# Load testing with k6
k6 run tests/load-test.js
```

## 📝 Examples

### Synchronous Inference

```python
import requests

response = requests.post(
    "http://localhost:8080/v1/infer",
    headers={"Authorization": "Bearer your-token"},
    json={
        "model": "gpt-4o",
        "prompt": "Hello, world!",
        "params": {
            "max_new_tokens": 128,
            "temperature": 0.7
        }
    }
)

print(response.json())
```

### Streaming Inference

```bash
curl -N -H "Authorization: Bearer token" \
  -H "Accept: text/event-stream" \
  -X POST http://localhost:8080/v1/infer/stream \
  -d '{"model":"gpt-4o","prompt":"Hello","params":{}}'
```

## 📚 Documentation

- [API Improvements](./INFERENCE_API_IMPROVEMENTS.md) - Detailed feature documentation
- [Architecture](./ARCHITECTURE.md) - System architecture overview
- [Deployment Guide](./DEPLOYMENT.md) - Production deployment guide

## 🛠️ Troubleshooting

### Common Issues

1. **Model not loading**: Check `TRUTHGPT_CONFIG` path and model files
2. **Redis connection failed**: Verify `REDIS_URL` and Redis availability
3. **High latency**: Check batch size and queue depth metrics
4. **Rate limit errors**: Adjust `RATE_LIMIT_RPM` or client request rate

### Debug Mode

```bash
export ENVIRONMENT=development
export LOG_LEVEL=DEBUG
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

See LICENSE file for details.

---

**Version:** 1.0.0  
**Status:** ✅ Production Ready


