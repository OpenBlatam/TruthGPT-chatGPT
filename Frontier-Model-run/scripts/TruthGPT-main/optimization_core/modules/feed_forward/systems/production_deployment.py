"""
Production Deployment Configuration
Complete production deployment setup for PiMoE systems including:
- Docker containerization
- Kubernetes deployment
- Monitoring and alerting
- Load balancing
- Auto-scaling
"""

import os
import json
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class DeploymentEnvironment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class ScalingStrategy(Enum):
    """Auto-scaling strategies."""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    HYBRID = "hybrid"

@dataclass
class DockerConfig:
    """Docker configuration for PiMoE system."""
    base_image: str = "pytorch/pytorch:2.0.1-cuda11.8-cudnn8-devel"
    python_version: str = "3.9"
    cuda_version: str = "11.8"
    working_dir: str = "/app"
    expose_port: int = 8080
    health_check_interval: int = 30
    health_check_timeout: int = 10
    health_check_retries: int = 3
    restart_policy: str = "unless-stopped"
    
    # Resource limits
    memory_limit: str = "8Gi"
    memory_reservation: str = "4Gi"
    cpu_limit: str = "4"
    cpu_reservation: str = "2"
    
    # Environment variables
    environment_vars: Dict[str, str] = None
    
    def __post_init__(self):
        if self.environment_vars is None:
            self.environment_vars = {
                "PYTHONPATH": "/app",
                "TORCH_CUDA_ARCH_LIST": "6.0;6.1;7.0;7.5;8.0;8.6",
                "CUDA_VISIBLE_DEVICES": "0"
            }

@dataclass
class KubernetesConfig:
    """Kubernetes configuration for PiMoE system."""
    namespace: str = "pimoe-production"
    service_name: str = "pimoe-service"
    deployment_name: str = "pimoe-deployment"
    replicas: int = 3
    min_replicas: int = 2
    max_replicas: int = 10
    
    # Resource requirements
    cpu_request: str = "2"
    cpu_limit: str = "4"
    memory_request: str = "4Gi"
    memory_limit: str = "8Gi"
    
    # Auto-scaling
    enable_hpa: bool = True
    target_cpu_utilization: int = 70
    target_memory_utilization: int = 80
    
    # Service configuration
    service_type: str = "ClusterIP"
    service_port: int = 8080
    target_port: int = 8080
    
    # Ingress configuration
    enable_ingress: bool = True
    ingress_host: str = "pimoe.example.com"
    ingress_path: str = "/api/v1"
    tls_secret: str = "pimoe-tls"

@dataclass
class MonitoringConfig:
    """Monitoring configuration for PiMoE system."""
    enable_prometheus: bool = True
    enable_grafana: bool = True
    enable_jaeger: bool = True
    enable_elasticsearch: bool = True
    
    # Prometheus configuration
    prometheus_port: int = 9090
    scrape_interval: str = "15s"
    evaluation_interval: str = "15s"
    
    # Grafana configuration
    grafana_port: int = 3000
    admin_password: str = "admin123"
    
    # Alerting rules
    alert_rules: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.alert_rules is None:
            self.alert_rules = [
                {
                    "alert": "HighCPUUsage",
                    "expr": "cpu_usage_percent > 80",
                    "for": "5m",
                    "labels": {"severity": "warning"},
                    "annotations": {"summary": "High CPU usage detected"}
                },
                {
                    "alert": "HighMemoryUsage",
                    "expr": "memory_usage_percent > 85",
                    "for": "5m",
                    "labels": {"severity": "warning"},
                    "annotations": {"summary": "High memory usage detected"}
                },
                {
                    "alert": "HighErrorRate",
                    "expr": "error_rate > 0.1",
                    "for": "2m",
                    "labels": {"severity": "critical"},
                    "annotations": {"summary": "High error rate detected"}
                }
            ]

@dataclass
class LoadBalancerConfig:
    """Load balancer configuration."""
    algorithm: str = "round_robin"  # round_robin, least_connections, ip_hash
    health_check_path: str = "/health"
    health_check_interval: int = 30
    health_check_timeout: int = 5
    max_retries: int = 3
    
    # SSL/TLS configuration
    enable_ssl: bool = True
    ssl_cert_path: str = "/etc/ssl/certs/pimoe.crt"
    ssl_key_path: str = "/etc/ssl/private/pimoe.key"
    
    # Rate limiting
    enable_rate_limiting: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds

class ProductionDeployment:
    """Production deployment manager for PiMoE systems."""
    
    def __init__(self, environment: DeploymentEnvironment = DeploymentEnvironment.PRODUCTION):
        self.environment = environment
        self.docker_config = DockerConfig()
        self.k8s_config = KubernetesConfig()
        self.monitoring_config = MonitoringConfig()
        self.load_balancer_config = LoadBalancerConfig()
    
    def generate_dockerfile(self) -> str:
        """Generate Dockerfile for PiMoE system."""
        dockerfile = f"""
# PiMoE Production Dockerfile
FROM {self.docker_config.base_image}

# Set working directory
WORKDIR {self.docker_config.working_dir}

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    wget \\
    git \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH={self.docker_config.working_dir}
ENV TORCH_CUDA_ARCH_LIST=6.0;6.1;7.0;7.5;8.0;8.6
ENV CUDA_VISIBLE_DEVICES=0

# Expose port
EXPOSE {self.docker_config.expose_port}

# Health check
HEALTHCHECK --interval={self.docker_config.health_check_interval}s \\
    --timeout={self.docker_config.health_check_timeout}s \\
    --retries={self.docker_config.health_check_retries} \\
    CMD curl -f http://localhost:{self.docker_config.expose_port}/health || exit 1

# Start application
CMD ["python", "production_pimoe_system.py"]
"""
        return dockerfile.strip()
    
    def generate_docker_compose(self) -> str:
        """Generate docker-compose.yml for PiMoE system."""
        compose = f"""
version: '3.8'

services:
  pimoe-service:
    build: .
    ports:
      - "8080:{self.docker_config.expose_port}"
    environment:
      - PYTHONPATH={self.docker_config.working_dir}
      - TORCH_CUDA_ARCH_LIST=6.0;6.1;7.0;7.5;8.0;8.6
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        limits:
          memory: {self.docker_config.memory_limit}
          cpus: '{self.docker_config.cpu_limit}'
        reservations:
          memory: {self.docker_config.memory_reservation}
          cpus: '{self.docker_config.cpu_reservation}'
    restart: {self.docker_config.restart_policy}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{self.docker_config.expose_port}/health"]
      interval: {self.docker_config.health_check_interval}s
      timeout: {self.docker_config.health_check_timeout}s
      retries: {self.docker_config.health_check_retries}
      start_period: 40s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - {self.load_balancer_config.ssl_cert_path}:/etc/ssl/certs/pimoe.crt
      - {self.load_balancer_config.ssl_key_path}:/etc/ssl/private/pimoe.key
    depends_on:
      - pimoe-service
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:{self.monitoring_config.prometheus_port}"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:{self.monitoring_config.grafana_port}"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD={self.monitoring_config.admin_password}
    volumes:
      - grafana-storage:/var/lib/grafana
    restart: unless-stopped

volumes:
  grafana-storage:
"""
        return compose.strip()
    
    def generate_kubernetes_manifests(self) -> Dict[str, str]:
        """Generate Kubernetes manifests for PiMoE system."""
        manifests = {}
        
        # Namespace
        manifests['namespace.yaml'] = f"""
apiVersion: v1
kind: Namespace
metadata:
  name: {self.k8s_config.namespace}
  labels:
    name: {self.k8s_config.namespace}
"""
        
        # Deployment
        manifests['deployment.yaml'] = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {self.k8s_config.deployment_name}
  namespace: {self.k8s_config.namespace}
  labels:
    app: pimoe
spec:
  replicas: {self.k8s_config.replicas}
  selector:
    matchLabels:
      app: pimoe
  template:
    metadata:
      labels:
        app: pimoe
    spec:
      containers:
      - name: pimoe
        image: pimoe:latest
        ports:
        - containerPort: {self.k8s_config.target_port}
        resources:
          requests:
            memory: "{self.k8s_config.memory_request}"
            cpu: "{self.k8s_config.cpu_request}"
          limits:
            memory: "{self.k8s_config.memory_limit}"
            cpu: "{self.k8s_config.cpu_limit}"
        env:
        - name: PYTHONPATH
          value: "/app"
        - name: TORCH_CUDA_ARCH_LIST
          value: "6.0;6.1;7.0;7.5;8.0;8.6"
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        livenessProbe:
          httpGet:
            path: /health
            port: {self.k8s_config.target_port}
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: {self.k8s_config.target_port}
          initialDelaySeconds: 5
          periodSeconds: 5
"""
        
        # Service
        manifests['service.yaml'] = f"""
apiVersion: v1
kind: Service
metadata:
  name: {self.k8s_config.service_name}
  namespace: {self.k8s_config.namespace}
  labels:
    app: pimoe
spec:
  type: {self.k8s_config.service_type}
  ports:
  - port: {self.k8s_config.service_port}
    targetPort: {self.k8s_config.target_port}
    protocol: TCP
  selector:
    app: pimoe
"""
        
        # Horizontal Pod Autoscaler
        if self.k8s_config.enable_hpa:
            manifests['hpa.yaml'] = f"""
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pimoe-hpa
  namespace: {self.k8s_config.namespace}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {self.k8s_config.deployment_name}
  minReplicas: {self.k8s_config.min_replicas}
  maxReplicas: {self.k8s_config.max_replicas}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: {self.k8s_config.target_cpu_utilization}
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: {self.k8s_config.target_memory_utilization}
"""
        
        # Ingress
        if self.k8s_config.enable_ingress:
            manifests['ingress.yaml'] = f"""
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: pimoe-ingress
  namespace: {self.k8s_config.namespace}
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - {self.k8s_config.ingress_host}
    secretName: {self.k8s_config.tls_secret}
  rules:
  - host: {self.k8s_config.ingress_host}
    http:
      paths:
      - path: {self.k8s_config.ingress_path}
        pathType: Prefix
        backend:
          service:
            name: {self.k8s_config.service_name}
            port:
              number: {self.k8s_config.service_port}
"""
        
        return manifests
    
    def generate_nginx_config(self) -> str:
        """Generate Nginx configuration for load balancing."""
        nginx_config = f"""
events {{
    worker_connections 1024;
}}

http {{
    upstream pimoe_backend {{
        least_conn;
        server pimoe-service:8080 max_fails=3 fail_timeout=30s;
        # Add more servers for horizontal scaling
    }}
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    server {{
        listen 80;
        server_name {self.load_balancer_config.ingress_host};
        
        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }}
    
    server {{
        listen 443 ssl http2;
        server_name {self.load_balancer_config.ingress_host};
        
        # SSL configuration
        ssl_certificate {self.load_balancer_config.ssl_cert_path};
        ssl_certificate_key {self.load_balancer_config.ssl_key_path};
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        
        # Rate limiting
        limit_req zone=api burst=20 nodelay;
        
        # Health check endpoint
        location /health {{
            access_log off;
            return 200 "healthy\\n";
            add_header Content-Type text/plain;
        }}
        
        # API endpoints
        location /api/v1/ {{
            proxy_pass http://pimoe_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
            
            # Buffer settings
            proxy_buffering on;
            proxy_buffer_size 4k;
            proxy_buffers 8 4k;
        }}
    }}
}}
"""
        return nginx_config
    
    def generate_prometheus_config(self) -> str:
        """Generate Prometheus configuration."""
        prometheus_config = f"""
global:
  scrape_interval: {self.monitoring_config.scrape_interval}
  evaluation_interval: {self.monitoring_config.evaluation_interval}

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'pimoe'
    static_configs:
      - targets: ['pimoe-service:8080']
    metrics_path: /metrics
    scrape_interval: 15s
    
  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:9113']
    scrape_interval: 15s
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
    scrape_interval: 15s
"""
        return prometheus_config
    
    def generate_grafana_dashboard(self) -> Dict[str, Any]:
        """Generate Grafana dashboard configuration."""
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "PiMoE Production Dashboard",
                "tags": ["pimoe", "production"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "Request Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(http_requests_total[5m])",
                                "legendFormat": "{{instance}}"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Requests/sec"
                            }
                        ]
                    },
                    {
                        "id": 2,
                        "title": "Response Time",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
                                "legendFormat": "95th percentile"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Seconds"
                            }
                        ]
                    },
                    {
                        "id": 3,
                        "title": "Error Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
                                "legendFormat": "5xx errors"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Errors/sec"
                            }
                        ]
                    },
                    {
                        "id": 4,
                        "title": "CPU Usage",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(process_cpu_seconds_total[5m]) * 100",
                                "legendFormat": "CPU %"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Percentage"
                            }
                        ]
                    },
                    {
                        "id": 5,
                        "title": "Memory Usage",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "process_resident_memory_bytes / 1024 / 1024",
                                "legendFormat": "Memory MB"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "MB"
                            }
                        ]
                    }
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "30s"
            }
        }
        return dashboard
    
    def generate_requirements(self) -> str:
        """Generate requirements.txt for production."""
        requirements = """
# Core dependencies
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
numpy>=1.21.0
scipy>=1.7.0

# Production dependencies
fastapi>=0.100.0
uvicorn[standard]>=0.20.0
pydantic>=2.0.0
python-multipart>=0.0.6

# Monitoring and logging
prometheus-client>=0.17.0
structlog>=23.0.0
python-json-logger>=2.0.0

# System monitoring
psutil>=5.9.0
py-cpuinfo>=9.0.0

# HTTP client
httpx>=0.24.0
aiohttp>=3.8.0

# Database (optional)
sqlalchemy>=2.0.0
alembic>=1.11.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0

# Development
black>=23.0.0
flake8>=6.0.0
mypy>=1.4.0
"""
        return requirements.strip()
    
    def generate_deployment_scripts(self) -> Dict[str, str]:
        """Generate deployment scripts."""
        scripts = {}
        
        # Build script
        scripts['build.sh'] = """#!/bin/bash
# Build PiMoE production image

set -e

echo "Building PiMoE production image..."

# Build Docker image
docker build -t pimoe:latest .

# Tag for registry
docker tag pimoe:latest pimoe:$(date +%Y%m%d-%H%M%S)

echo "Build completed successfully!"
"""
        
        # Deploy script
        scripts['deploy.sh'] = f"""#!/bin/bash
# Deploy PiMoE to Kubernetes

set -e

echo "Deploying PiMoE to Kubernetes..."

# Create namespace
kubectl apply -f namespace.yaml

# Deploy application
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Deploy HPA if enabled
if [ "{self.k8s_config.enable_hpa}" = "true" ]; then
    kubectl apply -f hpa.yaml
fi

# Deploy ingress if enabled
if [ "{self.k8s_config.enable_ingress}" = "true" ]; then
    kubectl apply -f ingress.yaml
fi

# Wait for deployment
kubectl rollout status deployment/{self.k8s_config.deployment_name} -n {self.k8s_config.namespace}

echo "Deployment completed successfully!"
"""
        
        # Health check script
        scripts['health_check.sh'] = """#!/bin/bash
# Health check script for PiMoE

set -e

HEALTH_URL="http://localhost:8080/health"
MAX_RETRIES=5
RETRY_DELAY=10

echo "Checking PiMoE health..."

for i in $(seq 1 $MAX_RETRIES); do
    if curl -f $HEALTH_URL > /dev/null 2>&1; then
        echo "✅ PiMoE is healthy!"
        exit 0
    else
        echo "❌ Health check failed (attempt $i/$MAX_RETRIES)"
        sleep $RETRY_DELAY
    fi
done

echo "❌ Health check failed after $MAX_RETRIES attempts"
exit 1
"""
        
        # Monitoring setup script
        scripts['setup_monitoring.sh'] = """#!/bin/bash
# Setup monitoring stack

set -e

echo "Setting up monitoring stack..."

# Deploy Prometheus
kubectl apply -f prometheus-deployment.yaml
kubectl apply -f prometheus-service.yaml

# Deploy Grafana
kubectl apply -f grafana-deployment.yaml
kubectl apply -f grafana-service.yaml

# Deploy Node Exporter
kubectl apply -f node-exporter-daemonset.yaml

echo "Monitoring stack deployed successfully!"
"""
        
        return scripts
    
    def save_deployment_files(self, output_dir: str = "deployment"):
        """Save all deployment files to directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save Docker files
        with open(os.path.join(output_dir, "Dockerfile"), "w") as f:
            f.write(self.generate_dockerfile())
        
        with open(os.path.join(output_dir, "docker-compose.yml"), "w") as f:
            f.write(self.generate_docker_compose())
        
        # Save Kubernetes manifests
        k8s_manifests = self.generate_kubernetes_manifests()
        for filename, content in k8s_manifests.items():
            with open(os.path.join(output_dir, filename), "w") as f:
                f.write(content)
        
        # Save configuration files
        with open(os.path.join(output_dir, "nginx.conf"), "w") as f:
            f.write(self.generate_nginx_config())
        
        with open(os.path.join(output_dir, "prometheus.yml"), "w") as f:
            f.write(self.generate_prometheus_config())
        
        with open(os.path.join(output_dir, "grafana-dashboard.json"), "w") as f:
            json.dump(self.generate_grafana_dashboard(), f, indent=2)
        
        with open(os.path.join(output_dir, "requirements.txt"), "w") as f:
            f.write(self.generate_requirements())
        
        # Save deployment scripts
        scripts = self.generate_deployment_scripts()
        for filename, content in scripts.items():
            script_path = os.path.join(output_dir, filename)
            with open(script_path, "w") as f:
                f.write(content)
            os.chmod(script_path, 0o755)
        
        print(f"✅ Deployment files saved to {output_dir}/")

def create_production_deployment(
    environment: DeploymentEnvironment = DeploymentEnvironment.PRODUCTION,
    **kwargs
) -> ProductionDeployment:
    """
    Factory function to create production deployment configuration.
    """
    deployment = ProductionDeployment(environment)
    
    # Update configurations with kwargs
    if 'docker_config' in kwargs:
        for key, value in kwargs['docker_config'].items():
            setattr(deployment.docker_config, key, value)
    
    if 'k8s_config' in kwargs:
        for key, value in kwargs['k8s_config'].items():
            setattr(deployment.k8s_config, key, value)
    
    if 'monitoring_config' in kwargs:
        for key, value in kwargs['monitoring_config'].items():
            setattr(deployment.monitoring_config, key, value)
    
    if 'load_balancer_config' in kwargs:
        for key, value in kwargs['load_balancer_config'].items():
            setattr(deployment.load_balancer_config, key, value)
    
    return deployment

def run_production_deployment_demo():
    """Run production deployment demonstration."""
    print("🚀 Production Deployment Demo")
    print("=" * 50)
    
    # Create production deployment
    deployment = create_production_deployment(
        environment=DeploymentEnvironment.PRODUCTION,
        k8s_config={
            'replicas': 3,
            'min_replicas': 2,
            'max_replicas': 10,
            'enable_hpa': True,
            'enable_ingress': True
        },
        monitoring_config={
            'enable_prometheus': True,
            'enable_grafana': True,
            'prometheus_port': 9090,
            'grafana_port': 3000
        }
    )
    
    print(f"📋 Deployment Configuration:")
    print(f"  Environment: {deployment.environment.value}")
    print(f"  Namespace: {deployment.k8s_config.namespace}")
    print(f"  Replicas: {deployment.k8s_config.replicas}")
    print(f"  Min Replicas: {deployment.k8s_config.min_replicas}")
    print(f"  Max Replicas: {deployment.k8s_config.max_replicas}")
    print(f"  HPA Enabled: {deployment.k8s_config.enable_hpa}")
    print(f"  Ingress Enabled: {deployment.k8s_config.enable_ingress}")
    
    print(f"\n🐳 Docker Configuration:")
    print(f"  Base Image: {deployment.docker_config.base_image}")
    print(f"  Expose Port: {deployment.docker_config.expose_port}")
    print(f"  Memory Limit: {deployment.docker_config.memory_limit}")
    print(f"  CPU Limit: {deployment.docker_config.cpu_limit}")
    
    print(f"\n📊 Monitoring Configuration:")
    print(f"  Prometheus: {deployment.monitoring_config.enable_prometheus}")
    print(f"  Grafana: {deployment.monitoring_config.enable_grafana}")
    print(f"  Prometheus Port: {deployment.monitoring_config.prometheus_port}")
    print(f"  Grafana Port: {deployment.monitoring_config.grafana_port}")
    
    # Generate deployment files
    print(f"\n📁 Generating Deployment Files...")
    deployment.save_deployment_files("pimoe_deployment")
    
    print(f"\n✅ Production deployment configuration completed!")
    print(f"📁 All deployment files saved to pimoe_deployment/")
    print(f"🚀 Ready for production deployment!")
    
    return deployment

if __name__ == "__main__":
    # Run production deployment demo
    deployment = run_production_deployment_demo()





