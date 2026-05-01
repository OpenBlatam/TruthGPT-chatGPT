# ⚡ Quick Start Guide

Guía rápida para empezar con la Inference API en 5 minutos.

## 🚀 Inicio Rápido (5 minutos)

### 1. Instalar Dependencias

```bash
pip install fastapi uvicorn httpx pydantic redis psutil
```

O para todas las dependencias:

```bash
pip install -r requirements_advanced.txt
```

### 2. Configurar Variables

```bash
export TRUTHGPT_API_TOKEN="your-secret-token"
export TRUTHGPT_CONFIG="configs/llm_default.yaml"
```

### 3. Iniciar API

```bash
python -m uvicorn inference.api:app --host 0.0.0.0 --port 8080
```

### 4. Probar la API

```bash
# Health check
curl http://localhost:8080/health

# Inferencia
curl -X POST http://localhost:8080/v1/infer \
  -H "Authorization: Bearer your-secret-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "prompt": "Hello, world!",
    "params": {
      "max_new_tokens": 64,
      "temperature": 0.7
    }
  }'
```

## 🐳 Docker (Aún Más Rápido)

### 1. Clonar/Descargar

```bash
cd agents/backend/onyx/server/features/Frontier-Model-run/scripts/TruthGPT-main/optimization_core/inference
```

### 2. Iniciar con Docker Compose

```bash
docker-compose up -d
```

### 3. Acceder

- **API**: http://localhost:8080
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Redis**: localhost:6379

## 🎯 Uso Básico

### Inferencia Síncrona

```python
import requests

response = requests.post(
    "http://localhost:8080/v1/infer",
    headers={"Authorization": "Bearer your-token"},
    json={
        "model": "gpt-4o",
        "prompt": "Explain quantum computing",
        "params": {
            "max_new_tokens": 128,
            "temperature": 0.7
        }
    }
)

print(response.json()["output"])
```

### Inferencia con Streaming

```bash
curl -N -H "Authorization: Bearer token" \
  -H "Accept: text/event-stream" \
  -X POST http://localhost:8080/v1/infer/stream \
  -d '{"model":"gpt-4o","prompt":"Hello","params":{}}'
```

### Ver Métricas

```bash
curl http://localhost:8080/metrics
```

### Health Check

```bash
curl http://localhost:8080/health
```

## 🛠️ Comandos Útiles

### Usando CLI

```bash
# Inferencia
python -m cli infer "Hello, world!" --max-tokens 128

# Servir API
python -m cli serve --port 8080

# Health check
python -m cli health

# Ver métricas
python -m cli metrics

# Testing
python -m cli test-api --iterations 10
```

### Usando Makefile

```bash
make run          # Desarrollo local
make test         # Tests
make docker-up    # Stack completo
make benchmark    # Performance testing
make tune         # Análisis de performance
make health       # Health check
```

## 📊 Ejemplos de Configuración

### Configuración Básica

```bash
# .env
TRUTHGPT_API_TOKEN=your-token
TRUTHGPT_CONFIG=configs/llm_default.yaml
PORT=8080
ENABLE_METRICS=true
ENABLE_TRACING=true
```

### Configuración Avanzada

```bash
# Batching optimizado
BATCH_MAX_SIZE=64
BATCH_FLUSH_TIMEOUT_MS=10

# Rate limiting
RATE_LIMIT_RPM=1000

# Caché Redis
CACHE_BACKEND=redis
REDIS_URL=redis://localhost:6379/0

# Observabilidad
OTLP_ENDPOINT=http://localhost:4317
```

## 🔍 Troubleshooting Rápido

### API no inicia
```bash
# Verificar puerto
netstat -an | grep 8080

# Verificar logs
python -m uvicorn inference.api:app --log-level debug
```

### Errores de autenticación
```bash
# Verificar token
echo $TRUTHGPT_API_TOKEN

# Probar con token explícito
curl -H "Authorization: Bearer your-token" http://localhost:8080/health
```

### Métricas no disponibles
```bash
# Verificar endpoint
curl http://localhost:8080/metrics

# Verificar configuración
echo $ENABLE_METRICS
```

## 📚 Próximos Pasos

1. **Lee el README completo**: `inference/README.md`
2. **Optimiza performance**: `inference/PERFORMANCE_GUIDE.md`
3. **Despliega a producción**: `DEPLOYMENT_COMPLETE.md`
4. **Configura alertas**: `prometheus/alerts.yml`

## 🎉 ¡Listo!

Ya tienes la API funcionando. Prueba diferentes endpoints y configura según tus necesidades.

---

**¿Necesitas ayuda?** Revisa la documentación completa o los ejemplos en los archivos de código.


