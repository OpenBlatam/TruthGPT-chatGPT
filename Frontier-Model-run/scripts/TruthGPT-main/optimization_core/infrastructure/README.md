# TruthGPT Infrastructure
# =======================

Complete, production-ready infrastructure for the TruthGPT Optimization Framework with enterprise-grade scalability, security, and monitoring.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        TruthGPT Infrastructure                  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Azure     │  │ Kubernetes  │  │ Monitoring  │            │
│  │   Cloud     │  │   Cluster    │  │   Stack     │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   CI/CD     │  │   Security   │  │   Storage   │            │
│  │  Pipelines  │  │   Policies   │  │   Systems   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Infrastructure Components

### 🚀 **Kubernetes Manifests**
- **Namespace & RBAC**: Secure multi-tenant isolation
- **Deployments**: Auto-scaling application pods
- **Services**: Load balancing and service discovery
- **ConfigMaps & Secrets**: Configuration management
- **Persistent Volumes**: Data persistence
- **Network Policies**: Micro-segmentation security
- **HPA/VPA**: Horizontal and vertical pod autoscaling

### 🔄 **Azure DevOps Pipelines**
- **CI/CD Pipeline**: Automated build, test, and deployment
- **Security Scanning**: SAST, DAST, and dependency scanning
- **Multi-Environment**: Dev, staging, and production deployments
- **Blue-Green Deployments**: Zero-downtime deployments
- **Rollback Capabilities**: Quick recovery from failures

### 🔧 **Ansible Automation**
- **Infrastructure Provisioning**: Automated server setup
- **Configuration Management**: Consistent environment configuration
- **Application Deployment**: Automated application deployment
- **Security Hardening**: Automated security configurations

### ☁️ **Terraform Infrastructure**
- **Azure Resources**: Complete cloud infrastructure
- **AKS Cluster**: Managed Kubernetes cluster
- **Networking**: VNet, subnets, and security groups
- **Storage**: Persistent storage solutions
- **Monitoring**: Log Analytics and Application Insights

### 📊 **Monitoring & Observability**
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **ELK Stack**: Log aggregation and analysis
- **Application Insights**: Application performance monitoring

### 🔒 **Security**
- **Network Policies**: Micro-segmentation
- **RBAC**: Role-based access control
- **Secrets Management**: Azure Key Vault integration
- **Security Scanning**: Automated vulnerability scanning

## 🚀 Quick Start

### Prerequisites

```bash
# Install required tools
az login
kubectl version --client
helm version
terraform version
ansible --version
```

### 1. Infrastructure Setup

```bash
# Navigate to infrastructure directory
cd infrastructure

# Initialize Terraform
cd terraform
terraform init
terraform plan
terraform apply

# Get AKS credentials
az aks get-credentials --resource-group truthgpt-production-rg --name truthgpt-aks
```

### 2. Deploy TruthGPT

```bash
# Using Helm (Recommended)
cd helm
helm install truthgpt ./truthgpt \
  --namespace truthgpt-optimization \
  --create-namespace \
  --values values.yaml

# Using Kubernetes manifests
kubectl apply -f kubernetes/
```

### 3. Deploy Monitoring

```bash
# Deploy Prometheus
helm install prometheus prometheus-community/prometheus \
  --namespace monitoring \
  --create-namespace

# Deploy Grafana
helm install grafana grafana/grafana \
  --namespace monitoring \
  --set adminPassword=admin
```

## 🔧 Configuration

### Environment Variables

```bash
# Required environment variables
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_RESOURCE_GROUP="truthgpt-production-rg"
export AKS_CLUSTER_NAME="truthgpt-aks"
export SECRET_KEY="your-secret-key"
export JWT_SECRET="your-jwt-secret"
export REDIS_PASSWORD="your-redis-password"
export WANDB_API_KEY="your-wandb-key"
export AZURE_STORAGE_KEY="your-storage-key"
```

### Helm Values

```yaml
# Customize deployment
app:
  replicaCount: 5
  resources:
    requests:
      memory: "4Gi"
      cpu: "2000m"
    limits:
      memory: "16Gi"
      cpu: "8000m"

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 50
```

## 📊 Monitoring

### Access Dashboards

```bash
# Get Grafana admin password
kubectl get secret --namespace monitoring grafana -o jsonpath="{.data.admin-password}" | base64 --decode

# Port forward to access Grafana
kubectl port-forward --namespace monitoring svc/grafana 3000:80

# Access Grafana at http://localhost:3000
# Username: admin
# Password: [from above command]
```

### Key Metrics

- **Application Metrics**: Request rate, response time, error rate
- **Resource Metrics**: CPU, memory, GPU utilization
- **Business Metrics**: Optimization tasks, success rate, throughput
- **Infrastructure Metrics**: Node health, pod status, network traffic

## 🔒 Security

### Network Policies

```yaml
# Example network policy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: truthgpt-network-policy
spec:
  podSelector:
    matchLabels:
      app: truthgpt-optimization
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
```

### RBAC Configuration

```yaml
# Service account with minimal permissions
apiVersion: v1
kind: ServiceAccount
metadata:
  name: truthgpt-service-account
  namespace: truthgpt-optimization
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: truthgpt-role
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
```

## 🚀 Scaling

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: truthgpt-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: truthgpt-optimization
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Vertical Pod Autoscaler

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: truthgpt-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: truthgpt-optimization
  updatePolicy:
    updateMode: "Auto"
```

## 🔄 CI/CD Pipeline

### Azure DevOps Pipeline

```yaml
# azure-pipelines.yml
trigger:
  branches:
    include:
    - main
    - develop

stages:
- stage: Build
  jobs:
  - job: BuildJob
    steps:
    - script: |
        echo "Building TruthGPT application..."
        docker build -t truthgpt/optimization:$(Build.BuildId) .
        docker push truthgpt/optimization:$(Build.BuildId)
```

### Deployment Strategies

1. **Blue-Green Deployment**: Zero-downtime deployments
2. **Canary Deployment**: Gradual rollout with monitoring
3. **Rolling Update**: Default Kubernetes deployment strategy

## 📈 Performance Optimization

### Resource Optimization

```yaml
# Optimized resource configuration
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
    nvidia.com/gpu: "1"
  limits:
    memory: "8Gi"
    cpu: "4000m"
    nvidia.com/gpu: "1"
```

### Caching Strategy

- **Redis**: In-memory caching for session data
- **CDN**: Static content delivery
- **Application Cache**: Model and data caching

## 🛠️ Troubleshooting

### Common Issues

1. **Pod Startup Issues**
   ```bash
   kubectl describe pod -n truthgpt-optimization
   kubectl logs -n truthgpt-optimization -l app=truthgpt-optimization
   ```

2. **Resource Constraints**
   ```bash
   kubectl top pods -n truthgpt-optimization
   kubectl top nodes
   ```

3. **Network Issues**
   ```bash
   kubectl get networkpolicies -n truthgpt-optimization
   kubectl describe networkpolicy -n truthgpt-optimization
   ```

### Health Checks

```bash
# Application health
curl http://truthgpt-api.example.com/health

# Kubernetes health
kubectl get all -n truthgpt-optimization

# Monitoring health
kubectl get pods -n monitoring
```

## 📚 Documentation

### Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Azure AKS Documentation](https://docs.microsoft.com/en-us/azure/aks/)
- [Helm Documentation](https://helm.sh/docs/)
- [Terraform Azure Provider](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)

### Support

- **Issues**: [GitHub Issues](https://github.com/truthgpt/optimization/issues)
- **Discord**: [TruthGPT Community](https://discord.gg/truthgpt)
- **Email**: infrastructure@truthgpt.ai

## 🏷️ Versioning

- **Infrastructure Version**: 3.0.0
- **Kubernetes Version**: 1.28+
- **Helm Version**: 3.0+
- **Terraform Version**: 1.0+

---

**Built with ❤️ by the TruthGPT Infrastructure Team**


