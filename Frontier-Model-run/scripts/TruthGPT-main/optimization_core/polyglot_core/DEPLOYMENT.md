# 🚀 Polyglot Core - Deployment Guide

Complete guide for deploying polyglot_core in different environments.

## 📦 Installation

### Production Installation

```bash
# Install with all backends
pip install optimization-core[all]

# Or install specific backends
pip install optimization-core[rust,cpp]  # Rust + C++ only
```

### From Source

```bash
# Clone repository
git clone https://github.com/truthgpt/optimization_core.git
cd optimization_core

# Build Rust backend
cd rust_core
maturin build --release
pip install target/wheels/*.whl
cd ..

# Build C++ backend
cd cpp_core
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
pip install .
cd ../..
```

## ⚙️ Configuration

### Environment Variables

```bash
# Set environment
export POLYGLOT_ENV=production
export POLYGLOT_DEBUG=false
export POLYGLOT_LOG_LEVEL=WARNING

# Performance
export POLYGLOT_ENABLE_PROFILING=true
export POLYGLOT_ENABLE_METRICS=true

# Resources
export POLYGLOT_MAX_MEMORY_GB=128
```

### Configuration File

```yaml
# ~/.polyglot_core/config.yaml
environment: production
debug: false
log_level: WARNING

preferred_backends:
  kv_cache: rust
  attention: cpp
  compression: rust

performance:
  enable_profiling: true
  enable_metrics: true

resources:
  max_memory_gb: 128
  max_cache_size_gb: 64
```

Load configuration:

```python
from optimization_core.polyglot_core import load_config

config = load_config("config.yaml")
```

## 🐳 Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (for rust_core)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install optimization_core
COPY . /app
WORKDIR /app
RUN pip install -e .

# Set environment
ENV POLYGLOT_ENV=production
ENV POLYGLOT_ENABLE_PROFILING=true

CMD ["python", "-m", "optimization_core.polyglot_core.scripts.check_backends"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  polyglot-core:
    build: .
    environment:
      - POLYGLOT_ENV=production
      - POLYGLOT_MAX_MEMORY_GB=64
    volumes:
      - ./config:/app/config
    ports:
      - "8080:8080"
```

## ☸️ Kubernetes Deployment

### Deployment YAML

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: polyglot-core
spec:
  replicas: 3
  selector:
    matchLabels:
      app: polyglot-core
  template:
    metadata:
      labels:
        app: polyglot-core
    spec:
      containers:
      - name: polyglot-core
        image: polyglot-core:latest
        env:
        - name: POLYGLOT_ENV
          value: "production"
        - name: POLYGLOT_MAX_MEMORY_GB
          value: "64"
        resources:
          requests:
            memory: "32Gi"
            cpu: "8"
          limits:
            memory: "64Gi"
            cpu: "16"
        volumeMounts:
        - name: config
          mountPath: /app/config
      volumes:
      - name: config
        configMap:
          name: polyglot-config
```

### ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: polyglot-config
data:
  config.yaml: |
    environment: production
    debug: false
    preferred_backends:
      kv_cache: rust
      attention: cpp
```

## 📊 Monitoring

### Metrics Collection

```python
from optimization_core.polyglot_core import get_metrics_collector

collector = get_metrics_collector()

# Metrics are automatically collected
# Export to Prometheus format
metrics_json = collector.export_json()
```

### Profiling

```python
from optimization_core.polyglot_core import get_profiler

profiler = get_profiler()
profiler.start_monitoring(interval=5.0)

# ... run operations ...

profiler.stop_monitoring()
profiler.print_summary()
```

## 🔍 Health Checks

### Backend Check Script

```bash
python -m optimization_core.polyglot_core.scripts.check_backends
```

### Programmatic Check

```python
from optimization_core.polyglot_core import (
    check_polyglot_availability,
    get_test_compatibility_info
)

availability = check_polyglot_availability()
compatibility = get_test_compatibility_info()

if not compatibility['polyglot_available']:
    raise RuntimeError("Polyglot core not available")
```

## 🚨 Troubleshooting

### Backend Not Available

```python
from optimization_core.polyglot_core import print_backend_status

print_backend_status()  # Check what's available
```

### Memory Issues

```python
from optimization_core.polyglot_core import get_config

config = get_config()
config.resources['max_memory_gb'] = 32  # Reduce if needed
```

### Performance Issues

```python
# Enable profiling
from optimization_core.polyglot_core import get_profiler

profiler = get_profiler()
with profiler.profile("slow_operation"):
    # Your code
    pass

profiler.print_summary()
```

## 📈 Production Best Practices

1. **Use Production Config**: Always use `PolyglotConfig.production()`
2. **Enable Metrics**: Set `enable_metrics: true`
3. **Monitor Resources**: Track memory and CPU usage
4. **Use Preferred Backends**: Configure backend preferences
5. **Set Resource Limits**: Configure max memory and cache size
6. **Enable Profiling**: For performance monitoring
7. **Health Checks**: Regular backend availability checks

## 🔐 Security

- Set `debug: false` in production
- Use `log_level: WARNING` or higher
- Limit resource usage
- Validate all inputs
- Use environment variables for secrets

---

For more information, see [README.md](README.md) and [API_REFERENCE.md](API_REFERENCE.md).












