# Guía de Despliegue - TruthGPT

Esta guía te llevará paso a paso para desplegar TruthGPT en diferentes entornos, desde desarrollo hasta producción.

## 📋 Tabla de Contenidos

1. [Despliegue Local](#despliegue-local)
2. [Despliegue en Docker](#despliegue-en-docker)
3. [Despliegue en la Nube](#despliegue-en-la-nube)
4. [Despliegue con Kubernetes](#despliegue-con-kubernetes)
5. [Despliegue con CI/CD](#despliegue-con-cicd)
6. [Monitoreo y Mantenimiento](#monitoreo-y-mantenimiento)

## 🏠 Despliegue Local

### Paso 1: Configuración del Entorno

```bash
# Crear entorno virtual
python -m venv truthgpt-env
source truthgpt-env/bin/activate  # En Windows: truthgpt-env\Scripts\activate

# Instalar dependencias
pip install -r requirements_modern.txt

# Verificar instalación
python -c "from optimization_core import *; print('✅ TruthGPT instalado correctamente')"
```

### Paso 2: Configuración Básica

```python
# config/local_config.py
from optimization_core import TruthGPTConfig

# Configuración para desarrollo local
LOCAL_CONFIG = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    use_mixed_precision=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=1,
    max_length=100,
    temperature=0.7
)

# Configuración de la API
API_CONFIG = {
    "host": "localhost",
    "port": 8000,
    "debug": True,
    "reload": True
}

# Configuración de la base de datos
DATABASE_CONFIG = {
    "type": "sqlite",
    "path": "truthgpt_local.db"
}
```

### Paso 3: Servidor Local

```python
# main.py
from fastapi import FastAPI
from optimization_core import ModernTruthGPTOptimizer, TruthGPTConfig
import uvicorn
from config.local_config import LOCAL_CONFIG, API_CONFIG

# Crear aplicación
app = FastAPI(
    title="TruthGPT Local",
    description="TruthGPT Optimization Core - Local Deployment",
    version="1.0.0"
)

# Inicializar optimizador
optimizer = ModernTruthGPTOptimizer(LOCAL_CONFIG)

@app.get("/")
async def root():
    return {"message": "TruthGPT Local Server", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "model": LOCAL_CONFIG.model_name}

@app.post("/generate")
async def generate_text(request: dict):
    try:
        text = request.get("text", "")
        max_length = request.get("max_length", 100)
        temperature = request.get("temperature", 0.7)
        
        generated = optimizer.generate(
            input_text=text,
            max_length=max_length,
            temperature=temperature
        )
        
        return {
            "generated_text": generated,
            "input_text": text,
            "parameters": {
                "max_length": max_length,
                "temperature": temperature
            }
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        debug=API_CONFIG["debug"],
        reload=API_CONFIG["reload"]
    )
```

### Paso 4: Ejecutar Servidor

```bash
# Ejecutar servidor
python main.py

# O con uvicorn directamente
uvicorn main:app --host localhost --port 8000 --reload
```

### Paso 5: Probar Despliegue

```python
# test_local_deployment.py
import requests
import json

def test_local_deployment():
    """Probar despliegue local"""
    base_url = "http://localhost:8000"
    
    # Probar salud
    response = requests.get(f"{base_url}/health")
    print(f"Health check: {response.json()}")
    
    # Probar generación
    payload = {
        "text": "Hola, ¿cómo estás?",
        "max_length": 100,
        "temperature": 0.7
    }
    
    response = requests.post(f"{base_url}/generate", json=payload)
    result = response.json()
    
    print(f"Generated text: {result['generated_text']}")
    print(f"Parameters: {result['parameters']}")

if __name__ == "__main__":
    test_local_deployment()
```

## 🐳 Despliegue en Docker

### Paso 1: Dockerfile

```dockerfile
# Dockerfile
FROM python:3.8-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements
COPY requirements_modern.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements_modern.txt

# Copiar código
COPY . .

# Crear usuario no-root
RUN useradd -m -u 1000 truthgpt && chown -R truthgpt:truthgpt /app
USER truthgpt

# Exponer puerto
EXPOSE 8000

# Comando de inicio
CMD ["python", "main.py"]
```

### Paso 2: Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  truthgpt:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME=microsoft/DialoGPT-medium
      - USE_MIXED_PRECISION=true
      - DEVICE=cuda
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    restart: unless-stopped
    
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
    
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
    restart: unless-stopped

volumes:
  grafana-storage:
```

### Paso 3: Configuración de Prometheus

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'truthgpt'
    static_configs:
      - targets: ['truthgpt:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s
```

### Paso 4: Desplegar con Docker

```bash
# Construir imagen
docker build -t truthgpt:latest .

# Ejecutar con Docker Compose
docker-compose up -d

# Verificar estado
docker-compose ps

# Ver logs
docker-compose logs -f truthgpt
```

### Paso 5: Probar Despliegue Docker

```python
# test_docker_deployment.py
import requests
import time

def test_docker_deployment():
    """Probar despliegue Docker"""
    base_url = "http://localhost:8000"
    
    # Esperar a que el servicio esté listo
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{base_url}/health")
            if response.status_code == 200:
                print("✅ Servicio Docker funcionando")
                break
        except requests.exceptions.ConnectionError:
            print(f"⏳ Esperando servicio... ({i+1}/{max_retries})")
            time.sleep(2)
    else:
        print("❌ Servicio no disponible")
        return
    
    # Probar generación
    payload = {
        "text": "Hola, ¿cómo estás?",
        "max_length": 100,
        "temperature": 0.7
    }
    
    response = requests.post(f"{base_url}/generate", json=payload)
    result = response.json()
    
    print(f"Generated text: {result['generated_text']}")

if __name__ == "__main__":
    test_docker_deployment()
```

## ☁️ Despliegue en la Nube

### AWS Deployment

#### Paso 1: Configuración AWS

```python
# config/aws_config.py
import boto3
from botocore.exceptions import ClientError

class AWSConfig:
    def __init__(self, region_name: str = 'us-east-1'):
        self.region_name = region_name
        self.ec2 = boto3.client('ec2', region_name=region_name)
        self.ecs = boto3.client('ecs', region_name=region_name)
        self.ecr = boto3.client('ecr', region_name=region_name)
        self.elb = boto3.client('elbv2', region_name=region_name)
    
    def create_ecr_repository(self, repository_name: str):
        """Crear repositorio ECR"""
        try:
            response = self.ecr.create_repository(
                repositoryName=repository_name
            )
            print(f"✅ Repositorio ECR creado: {repository_name}")
            return response
        except ClientError as e:
            if e.response['Error']['Code'] == 'RepositoryAlreadyExistsException':
                print(f"📦 Repositorio ECR ya existe: {repository_name}")
            else:
                print(f"❌ Error al crear repositorio: {e}")
    
    def push_image_to_ecr(self, repository_name: str, image_tag: str):
        """Subir imagen a ECR"""
        try:
            # Obtener token de autenticación
            auth_token = self.ecr.get_authorization_token()
            username, password = auth_token['authorizationData'][0]['authorizationToken'].decode('base64').split(':')
            
            # Autenticar Docker
            import subprocess
            subprocess.run([
                'docker', 'login', '-u', username, '-p', password,
                auth_token['authorizationData'][0]['proxyEndpoint']
            ])
            
            # Etiquetar imagen
            subprocess.run([
                'docker', 'tag', f'{image_tag}:latest',
                f'{auth_token["authorizationData"][0]["proxyEndpoint"]}/{repository_name}:latest'
            ])
            
            # Subir imagen
            subprocess.run([
                'docker', 'push',
                f'{auth_token["authorizationData"][0]["proxyEndpoint"]}/{repository_name}:latest'
            ])
            
            print(f"✅ Imagen subida a ECR: {repository_name}")
            
        except Exception as e:
            print(f"❌ Error al subir imagen: {e}")
```

#### Paso 2: ECS Task Definition

```json
{
  "family": "truthgpt-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "truthgpt",
      "image": "ACCOUNT.dkr.ecr.REGION.amazonaws.com/truthgpt:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MODEL_NAME",
          "value": "microsoft/DialoGPT-medium"
        },
        {
          "name": "USE_MIXED_PRECISION",
          "value": "true"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/truthgpt",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Paso 3: ECS Service

```json
{
  "serviceName": "truthgpt-service",
  "cluster": "truthgpt-cluster",
  "taskDefinition": "truthgpt-task",
  "desiredCount": 2,
  "launchType": "FARGATE",
  "networkConfiguration": {
    "awsvpcConfiguration": {
      "subnets": ["subnet-12345", "subnet-67890"],
      "securityGroups": ["sg-12345"],
      "assignPublicIp": "ENABLED"
    }
  },
  "loadBalancers": [
    {
      "targetGroupArn": "arn:aws:elasticloadbalancing:REGION:ACCOUNT:targetgroup/truthgpt-tg/ID",
      "containerName": "truthgpt",
      "containerPort": 8000
    }
  ]
}
```

### Google Cloud Deployment

#### Paso 1: Configuración GCP

```python
# config/gcp_config.py
from google.cloud import run_v2
from google.cloud import container_v1
import os

class GCPConfig:
    def __init__(self, project_id: str, region: str = 'us-central1'):
        self.project_id = project_id
        self.region = region
        self.run_client = run_v2.ServicesClient()
        self.container_client = container_v1.ClusterManagerClient()
    
    def deploy_to_cloud_run(self, service_name: str, image_url: str):
        """Desplegar a Cloud Run"""
        try:
            service = run_v2.Service(
                template=run_v2.RevisionTemplate(
                    containers=[
                        run_v2.Container(
                            image=image_url,
                            ports=[run_v2.ContainerPort(container_port=8000)],
                            env=[
                                run_v2.EnvVar(name="MODEL_NAME", value="microsoft/DialoGPT-medium"),
                                run_v2.EnvVar(name="USE_MIXED_PRECISION", value="true")
                            ]
                        )
                    ]
                )
            )
            
            request = run_v2.CreateServiceRequest(
                parent=f"projects/{self.project_id}/locations/{self.region}",
                service=service,
                service_id=service_name
            )
            
            response = self.run_client.create_service(request=request)
            print(f"✅ Servicio desplegado en Cloud Run: {service_name}")
            return response
            
        except Exception as e:
            print(f"❌ Error al desplegar en Cloud Run: {e}")
    
    def deploy_to_gke(self, cluster_name: str, image_url: str):
        """Desplegar a Google Kubernetes Engine"""
        try:
            # Configurar kubectl
            os.system(f"gcloud container clusters get-credentials {cluster_name} --region {self.region}")
            
            # Crear deployment
            deployment_yaml = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: truthgpt
spec:
  replicas: 2
  selector:
    matchLabels:
      app: truthgpt
  template:
    metadata:
      labels:
        app: truthgpt
    spec:
      containers:
      - name: truthgpt
        image: {image_url}
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_NAME
          value: "microsoft/DialoGPT-medium"
        - name: USE_MIXED_PRECISION
          value: "true"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: truthgpt-service
spec:
  selector:
    app: truthgpt
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
"""
            
            with open("truthgpt-deployment.yaml", "w") as f:
                f.write(deployment_yaml)
            
            os.system("kubectl apply -f truthgpt-deployment.yaml")
            print(f"✅ Servicio desplegado en GKE: {cluster_name}")
            
        except Exception as e:
            print(f"❌ Error al desplegar en GKE: {e}")
```

## 🚀 Despliegue con Kubernetes

### Paso 1: Configuración de Kubernetes

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: truthgpt
```

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: truthgpt-config
  namespace: truthgpt
data:
  MODEL_NAME: "microsoft/DialoGPT-medium"
  USE_MIXED_PRECISION: "true"
  DEVICE: "cuda"
  BATCH_SIZE: "1"
  MAX_LENGTH: "100"
  TEMPERATURE: "0.7"
```

```yaml
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: truthgpt-secret
  namespace: truthgpt
type: Opaque
data:
  # Base64 encoded values
  API_KEY: <base64-encoded-api-key>
  DATABASE_URL: <base64-encoded-database-url>
```

### Paso 2: Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: truthgpt
  namespace: truthgpt
spec:
  replicas: 3
  selector:
    matchLabels:
      app: truthgpt
  template:
    metadata:
      labels:
        app: truthgpt
    spec:
      containers:
      - name: truthgpt
        image: truthgpt:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_NAME
          valueFrom:
            configMapKeyRef:
              name: truthgpt-config
              key: MODEL_NAME
        - name: USE_MIXED_PRECISION
          valueFrom:
            configMapKeyRef:
              name: truthgpt-config
              key: USE_MIXED_PRECISION
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: truthgpt-secret
              key: API_KEY
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Paso 3: Service

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: truthgpt-service
  namespace: truthgpt
spec:
  selector:
    app: truthgpt
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Paso 4: Ingress

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: truthgpt-ingress
  namespace: truthgpt
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - truthgpt.example.com
    secretName: truthgpt-tls
  rules:
  - host: truthgpt.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: truthgpt-service
            port:
              number: 80
```

### Paso 5: Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: truthgpt-hpa
  namespace: truthgpt
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: truthgpt
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Paso 6: Desplegar en Kubernetes

```bash
# Crear namespace
kubectl apply -f k8s/namespace.yaml

# Aplicar configuración
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml

# Desplegar aplicación
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml

# Verificar despliegue
kubectl get pods -n truthgpt
kubectl get services -n truthgpt
kubectl get ingress -n truthgpt
```

## 🔄 Despliegue con CI/CD

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy TruthGPT

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.8'
    
    - name: Install dependencies
      run: |
        pip install -r requirements_modern.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=optimization_core --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: |
          truthgpt:latest
          truthgpt:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    
    - name: Deploy to ECS
      run: |
        aws ecs update-service --cluster truthgpt-cluster --service truthgpt-service --force-new-deployment
    
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/truthgpt truthgpt=truthgpt:${{ github.sha }}
        kubectl rollout status deployment/truthgpt
```

### GitLab CI/CD

```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

variables:
  DOCKER_IMAGE: $CI_REGISTRY_IMAGE/truthgpt
  DOCKER_TAG: $CI_COMMIT_SHA

test:
  stage: test
  image: python:3.8
  script:
    - pip install -r requirements_modern.txt
    - pip install pytest pytest-cov
    - pytest tests/ --cov=optimization_core --cov-report=xml
  coverage: '/TOTAL.*\s+(\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $DOCKER_IMAGE:$DOCKER_TAG .
    - docker push $DOCKER_IMAGE:$DOCKER_TAG
    - docker tag $DOCKER_IMAGE:$DOCKER_TAG $DOCKER_IMAGE:latest
    - docker push $DOCKER_IMAGE:latest
  only:
    - main

deploy:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl set image deployment/truthgpt truthgpt=$DOCKER_IMAGE:$DOCKER_TAG
    - kubectl rollout status deployment/truthgpt
  only:
    - main
  when: manual
```

## 📊 Monitoreo y Mantenimiento

### Paso 1: Configuración de Monitoreo

```python
# monitoring/setup.py
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import time
import psutil
import torch

class TruthGPTMonitoring:
    def __init__(self, port: int = 9090):
        self.port = port
        
        # Métricas de Prometheus
        self.generation_counter = Counter(
            'truthgpt_generations_total',
            'Total number of text generations',
            ['model_name', 'status']
        )
        
        self.generation_duration = Histogram(
            'truthgpt_generation_duration_seconds',
            'Time spent on text generation',
            ['model_name']
        )
        
        self.memory_usage = Gauge(
            'truthgpt_memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        self.gpu_memory_usage = Gauge(
            'truthgpt_gpu_memory_usage_bytes',
            'GPU memory usage in bytes'
        )
        
        # Iniciar servidor de métricas
        start_http_server(self.port)
        print(f"📊 Servidor de métricas iniciado en puerto {self.port}")
    
    def update_system_metrics(self):
        """Actualizar métricas del sistema"""
        # Memoria del sistema
        memory = psutil.virtual_memory()
        self.memory_usage.set(memory.used)
        
        # Memoria GPU
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated()
            self.gpu_memory_usage.set(gpu_memory)
    
    def start_monitoring(self):
        """Iniciar monitoreo"""
        import threading
        
        def monitor_loop():
            while True:
                self.update_system_metrics()
                time.sleep(10)  # Actualizar cada 10 segundos
        
        monitor_thread = threading.Thread(target=monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        print("📊 Monitoreo iniciado")
```

### Paso 2: Alertas

```yaml
# monitoring/alerts.yml
groups:
- name: truthgpt
  rules:
  - alert: TruthGPTHighMemoryUsage
    expr: truthgpt_memory_usage_bytes > 4 * 1024 * 1024 * 1024  # 4GB
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "TruthGPT high memory usage"
      description: "TruthGPT memory usage is above 4GB for more than 5 minutes"
  
  - alert: TruthGPTHighGPUUsage
    expr: truthgpt_gpu_memory_usage_bytes > 8 * 1024 * 1024 * 1024  # 8GB
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "TruthGPT high GPU memory usage"
      description: "TruthGPT GPU memory usage is above 8GB for more than 5 minutes"
  
  - alert: TruthGPTGenerationFailure
    expr: rate(truthgpt_generations_total{status="error"}[5m]) > 0.1
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "TruthGPT generation failures"
      description: "TruthGPT generation failure rate is above 10% for more than 2 minutes"
```

### Paso 3: Dashboard de Grafana

```json
{
  "dashboard": {
    "title": "TruthGPT Monitoring Dashboard",
    "panels": [
      {
        "title": "Generations per Second",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(truthgpt_generations_total[5m])",
            "legendFormat": "Generations/sec"
          }
        ]
      },
      {
        "title": "Generation Duration",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(truthgpt_generation_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "truthgpt_memory_usage_bytes",
            "legendFormat": "Memory Usage"
          }
        ]
      },
      {
        "title": "GPU Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "truthgpt_gpu_memory_usage_bytes",
            "legendFormat": "GPU Memory Usage"
          }
        ]
      }
    ]
  }
}
```

### Paso 4: Mantenimiento Automático

```python
# maintenance/auto_maintenance.py
import schedule
import time
from optimization_core import ModernTruthGPTOptimizer, TruthGPTConfig

class TruthGPTAutoMaintenance:
    def __init__(self):
        self.config = TruthGPTConfig(
            model_name="microsoft/DialoGPT-medium",
            use_mixed_precision=True
        )
        self.optimizer = ModernTruthGPTOptimizer(self.config)
    
    def cleanup_old_data(self):
        """Limpiar datos antiguos"""
        print("🧹 Limpiando datos antiguos...")
        # Implementar limpieza de datos
        pass
    
    def optimize_model(self):
        """Optimizar modelo"""
        print("⚡ Optimizando modelo...")
        # Implementar optimización del modelo
        pass
    
    def backup_data(self):
        """Respaldar datos"""
        print("💾 Respaldando datos...")
        # Implementar respaldo de datos
        pass
    
    def health_check(self):
        """Verificar salud del sistema"""
        print("🏥 Verificando salud del sistema...")
        try:
            # Probar generación
            test_text = self.optimizer.generate(
                input_text="Test",
                max_length=10,
                temperature=0.7
            )
            print("✅ Sistema saludable")
        except Exception as e:
            print(f"❌ Error en verificación de salud: {e}")
    
    def start_maintenance_schedule(self):
        """Iniciar programación de mantenimiento"""
        # Limpiar datos cada día a las 2 AM
        schedule.every().day.at("02:00").do(self.cleanup_old_data)
        
        # Optimizar modelo cada semana
        schedule.every().week.do(self.optimize_model)
        
        # Respaldar datos cada día a las 3 AM
        schedule.every().day.at("03:00").do(self.backup_data)
        
        # Verificar salud cada hora
        schedule.every().hour.do(self.health_check)
        
        print("📅 Programación de mantenimiento iniciada")
        
        # Ejecutar programación
        while True:
            schedule.run_pending()
            time.sleep(60)  # Verificar cada minuto

# Iniciar mantenimiento automático
maintenance = TruthGPTAutoMaintenance()
maintenance.start_maintenance_schedule()
```

## 🎯 Próximos Pasos

### 1. Despliegue Multi-Región
```python
# Configuración para múltiples regiones
class MultiRegionDeployment:
    def __init__(self):
        self.regions = ['us-east-1', 'us-west-2', 'eu-west-1']
        self.configs = {}
        
        for region in self.regions:
            self.configs[region] = TruthGPTConfig(
                model_name="microsoft/DialoGPT-medium",
                use_mixed_precision=True,
                region=region
            )
    
    def deploy_to_all_regions(self):
        """Desplegar a todas las regiones"""
        for region in self.regions:
            print(f"🚀 Desplegando en {region}...")
            # Implementar despliegue por región
            pass
```

### 2. Despliegue con Blue-Green
```python
# Configuración Blue-Green
class BlueGreenDeployment:
    def __init__(self):
        self.blue_version = "v1.0.0"
        self.green_version = "v1.1.0"
        self.current_version = "blue"
    
    def switch_to_green(self):
        """Cambiar a versión verde"""
        print("🔄 Cambiando a versión verde...")
        # Implementar cambio de versión
        pass
    
    def rollback_to_blue(self):
        """Rollback a versión azul"""
        print("⏪ Rollback a versión azul...")
        # Implementar rollback
        pass
```

### 3. Despliegue con Canary
```python
# Configuración Canary
class CanaryDeployment:
    def __init__(self):
        self.canary_percentage = 10  # 10% del tráfico
        self.canary_version = "v1.1.0"
        self.stable_version = "v1.0.0"
    
    def deploy_canary(self):
        """Desplegar canary"""
        print(f"🐦 Desplegando canary con {self.canary_percentage}% del tráfico...")
        # Implementar despliegue canary
        pass
    
    def promote_canary(self):
        """Promover canary a estable"""
        print("🚀 Promoviendo canary a estable...")
        # Implementar promoción
        pass
```

---

*¡Con esta guía tienes todo lo necesario para desplegar TruthGPT en cualquier entorno! 🚀✨*


