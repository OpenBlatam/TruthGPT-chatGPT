# 🚀 Enterprise TruthGPT Quick Start Guide

Guía rápida para desplegar el sistema enterprise TruthGPT en Azure.

## ✅ Prerrequisitos

```bash
# Instalar herramientas necesarias
# Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Docker
sudo apt-get update && sudo apt-get install docker.io
sudo usermod -aG docker $USER

# Ansible
sudo apt-get install ansible
```

## 🎯 Despliegue Rápido

### 1. Configurar Azure

```bash
# Login a Azure
az login

# Crear resource group
az group create --name truthgpt-enterprise-rg --location "East US"

# Crear ACR
az acr create --resource-group truthgpt-enterprise-rg --name truthgptacr --sku Standard

# Crear AKS cluster
az aks create \
  --resource-group truthgpt-enterprise-rg \
  --name truthgpt-enterprise-cluster \
  --node-count 3 \
  --enable-addons monitoring \
  --enable-managed-identity \
  --enable-azure-rbac \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 10 \
  --attach-acr truthgptacr
```

### 2. Configurar kubectl

```bash
# Obtener credenciales
az aks get-credentials --resource-group truthgpt-enterprise-rg --name truthgpt-enterprise-cluster

# Verificar conexión
kubectl get nodes
```

### 3. Desplegar Application

```bash
# Opción 1: Usando script de Azure
./deployment/azure-deploy.sh

# Opción 2: Usando Ansible
ansible-playbook -i inventory/hosts deployment/ansible/playbooks/deploy-truthgpt.yml
```

### 4. Verificar Despliegue

```bash
# Ver pods
kubectl get pods -n truthgpt-enterprise

# Ver services
kubectl get services -n truthgpt-enterprise

# Ver ingress
kubectl get ingress -n truthgpt-enterprise

# Ver HPA
kubectl get hpa -n truthgpt-enterprise
```

## 📊 Monitoreo

```bash
# Monitoreo continuo
./deployment/scripts/enterprise-monitor.sh continuous

# Auto-scaling continuo
./deployment/scripts/enterprise-auto-scale.sh continuous

# Generar reporte
./deployment/scripts/enterprise-monitor.sh report
```

## 🎨 Acceso a la Aplicación

```bash
# Obtener URL del ingress
INGRESS_URL=$(kubectl get ingress -n truthgpt-enterprise -o jsonpath='{.items[0].status.loadBalancer.ingress[0].hostname}')
echo "Access URL: http://${INGRESS_URL}"

# Port-forward para acceso local
kubectl port-forward svc/truthgpt-enterprise-optimizer 8080:80 -n truthgpt-enterprise
# Access local: http://localhost:8080
```

## 🔧 Configuración

### Variables de Entorno

```bash
export AZURE_REGION="East US"
export AKS_CLUSTER_NAME="truthgpt-enterprise-cluster"
export NAMESPACE="truthgpt-enterprise"
export RESOURCE_GROUP="truthgpt-enterprise-rg"
export ACR_NAME="truthgptacr"
export OPTIMIZATION_LEVEL="enterprise_enterprise"
```

### Configuración Avanzada

Edita `deployment/config/config.yaml` para personalizar:

```yaml
application:
  name: "Enterprise TruthGPT"
  environment: "production"
  
optimization:
  level: "enterprise_enterprise"
  target_improvement: 1000000000000000000000.0
```

## 🚨 Troubleshooting

### Problemas Comunes

```bash
# Ver logs de pods
kubectl logs -n truthgpt-enterprise -l app=truthgpt-enterprise-optimizer --tail=100

# Ver eventos
kubectl get events -n truthgpt-enterprise --sort-by='.lastTimestamp'

# Ver descripción de pod
kubectl describe pod -n truthgpt-enterprise truthgpt-enterprise-optimizer-xxx

# Reiniciar deployment
kubectl rollout restart deployment/truthgpt-enterprise-optimizer -n truthgpt-enterprise
```

### Limpieza

```bash
# Limpiar recursos
./deployment/azure-deploy.sh --cleanup

# Eliminar cluster (opcional)
az aks delete --name truthgpt-enterprise-cluster --resource-group truthgpt-enterprise-rg
```

## 📚 Documentación

- [Configuración Completa](ENTERPRISE_README.md)
- [Guías de Optimización](documentation/guides/)
- [Ejemplos de Uso](documentation/examples/)
- [API Reference](documentation/api/)

## 🆘 Soporte

- **Email**: enterprise@truthgpt.com
- **Slack**: #truthgpt-enterprise
- **GitHub Issues**: https://github.com/truthgpt/enterprise-optimization/issues

---

**¡Desarrollado con ❤️ por el equipo de TruthGPT Enterprise!**







