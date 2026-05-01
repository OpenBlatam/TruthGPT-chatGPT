#!/bin/bash
# Enterprise TruthGPT Azure Deployment Script
# Deploys TruthGPT optimization system to Azure AKS with enterprise features

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_NAME="enterprise-truthgpt"
readonly AZURE_REGION="${AZURE_REGION:-East US}"
readonly AKS_CLUSTER_NAME="${AKS_CLUSTER_NAME:-truthgpt-enterprise-cluster}"
readonly NAMESPACE="${NAMESPACE:-truthgpt-enterprise}"
readonly ACR_NAME="${ACR_NAME:-truthgptacr}"
readonly IMAGE_TAG="${IMAGE_TAG:-latest}"
readonly RESOURCE_GROUP="${RESOURCE_GROUP:-truthgpt-enterprise-rg}"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Error handling
trap 'log_error "Script failed at line $LINENO"' ERR

# Check prerequisites
check_prerequisites() {
    log_info "Checking Azure prerequisites..."
    
    # Check Azure CLI
    if ! command -v az &> /dev/null; then
        log_error "Azure CLI not found. Please install it first."
        exit 1
    fi
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install it first."
        exit 1
    fi
    
    # Check helm
    if ! command -v helm &> /dev/null; then
        log_error "Helm not found. Please install it first."
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install it first."
        exit 1
    fi
    
    # Check Azure login
    if ! az account show &> /dev/null; then
        log_error "Not logged in to Azure. Please run 'az login'."
        exit 1
    fi
    
    log_success "All Azure prerequisites met"
}

# Create resource group
create_resource_group() {
    log_info "Creating Azure resource group: ${RESOURCE_GROUP}"
    
    # Check if resource group exists
    if az group show --name "${RESOURCE_GROUP}" &> /dev/null; then
        log_warning "Resource group ${RESOURCE_GROUP} already exists"
        return 0
    fi
    
    # Create resource group
    az group create \
        --name "${RESOURCE_GROUP}" \
        --location "${AZURE_REGION}" \
        --tags "Project=TruthGPT" "Environment=Enterprise" "Optimization=UltraSpeed"
    
    log_success "Resource group created successfully"
}

# Create Azure Container Registry
create_acr() {
    log_info "Creating Azure Container Registry: ${ACR_NAME}"
    
    # Check if ACR exists
    if az acr show --name "${ACR_NAME}" --resource-group "${RESOURCE_GROUP}" &> /dev/null; then
        log_warning "ACR ${ACR_NAME} already exists"
        return 0
    fi
    
    # Create ACR
    az acr create \
        --resource-group "${RESOURCE_GROUP}" \
        --name "${ACR_NAME}" \
        --sku Standard \
        --admin-enabled true \
        --location "${AZURE_REGION}"
    
    log_success "ACR created successfully"
}

# Create AKS cluster
create_aks_cluster() {
    log_info "Creating AKS cluster: ${AKS_CLUSTER_NAME}"
    
    # Check if cluster exists
    if az aks show --name "${AKS_CLUSTER_NAME}" --resource-group "${RESOURCE_GROUP}" &> /dev/null; then
        log_warning "AKS cluster ${AKS_CLUSTER_NAME} already exists"
        return 0
    fi
    
    # Create AKS cluster with enterprise configuration
    az aks create \
        --resource-group "${RESOURCE_GROUP}" \
        --name "${AKS_CLUSTER_NAME}" \
        --node-count 3 \
        --node-vm-size "Standard_D4s_v3" \
        --enable-addons monitoring \
        --enable-managed-identity \
        --enable-azure-rbac \
        --enable-cluster-autoscaler \
        --min-count 1 \
        --max-count 10 \
        --attach-acr "${ACR_NAME}" \
        --generate-ssh-keys \
        --tags "Project=TruthGPT" "Environment=Enterprise" "Optimization=UltraSpeed"
    
    log_success "AKS cluster created successfully"
}

# Configure kubectl
configure_kubectl() {
    log_info "Configuring kubectl for cluster: ${AKS_CLUSTER_NAME}"
    
    az aks get-credentials \
        --resource-group "${RESOURCE_GROUP}" \
        --name "${AKS_CLUSTER_NAME}" \
        --overwrite-existing
    
    # Verify cluster access
    if ! kubectl get nodes &> /dev/null; then
        log_error "Failed to access AKS cluster"
        exit 1
    fi
    
    log_success "kubectl configured successfully"
}

# Create namespace
create_namespace() {
    log_info "Creating namespace: ${NAMESPACE}"
    
    kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
    
    log_success "Namespace created successfully"
}

# Build and push Docker image
build_and_push_image() {
    log_info "Building and pushing Docker image..."
    
    # Login to ACR
    az acr login --name "${ACR_NAME}"
    
    # Build image
    docker build -t "${ACR_NAME}.azurecr.io/${PROJECT_NAME}:${IMAGE_TAG}" "${SCRIPT_DIR}/../"
    
    # Push image
    docker push "${ACR_NAME}.azurecr.io/${PROJECT_NAME}:${IMAGE_TAG}"
    
    # Tag as latest
    docker tag "${ACR_NAME}.azurecr.io/${PROJECT_NAME}:${IMAGE_TAG}" "${ACR_NAME}.azurecr.io/${PROJECT_NAME}:latest"
    docker push "${ACR_NAME}.azurecr.io/${PROJECT_NAME}:latest"
    
    log_success "Docker image built and pushed successfully"
}

# Deploy application
deploy_application() {
    log_info "Deploying TruthGPT enterprise optimization system..."
    
    # Apply ConfigMap
    kubectl apply -f "${SCRIPT_DIR}/k8s/configmap.yaml" -n "${NAMESPACE}"
    
    # Apply Secrets
    kubectl apply -f "${SCRIPT_DIR}/k8s/secrets.yaml" -n "${NAMESPACE}"
    
    # Apply Services
    kubectl apply -f "${SCRIPT_DIR}/k8s/services.yaml" -n "${NAMESPACE}"
    
    # Apply Deployments
    kubectl apply -f "${SCRIPT_DIR}/k8s/deployments.yaml" -n "${NAMESPACE}"
    
    # Apply HPA
    kubectl apply -f "${SCRIPT_DIR}/k8s/hpa.yaml" -n "${NAMESPACE}"
    
    # Apply Ingress
    kubectl apply -f "${SCRIPT_DIR}/k8s/ingress.yaml" -n "${NAMESPACE}"
    
    log_success "Application deployed successfully"
}

# Wait for deployment
wait_for_deployment() {
    log_info "Waiting for deployment to be ready..."
    
    kubectl wait --for=condition=available --timeout=300s deployment/truthgpt-enterprise-optimizer -n "${NAMESPACE}"
    
    log_success "Deployment is ready"
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up Azure monitoring..."
    
    # Install Prometheus
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    helm install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --set grafana.adminPassword=admin123 \
        --set prometheus.prometheusSpec.retention=30d \
        --set alertmanager.alertmanagerSpec.retention=30d
    
    # Install Grafana
    kubectl apply -f "${SCRIPT_DIR}/k8s/grafana-dashboard.yaml" -n monitoring
    
    log_success "Azure monitoring setup completed"
}

# Setup logging
setup_logging() {
    log_info "Setting up Azure logging..."
    
    # Install Fluentd
    kubectl apply -f "${SCRIPT_DIR}/k8s/fluentd.yaml" -n "${NAMESPACE}"
    
    # Install Elasticsearch
    helm repo add elastic https://helm.elastic.co
    helm repo update
    
    helm install elasticsearch elastic/elasticsearch \
        --namespace logging \
        --create-namespace \
        --set replicas=1 \
        --set volumeClaimTemplate.resources.requests.storage=10Gi
    
    # Install Kibana
    helm install kibana elastic/kibana \
        --namespace logging \
        --set elasticsearchHosts="http://elasticsearch-master:9200"
    
    log_success "Azure logging setup completed"
}

# Setup security
setup_security() {
    log_info "Setting up Azure security..."
    
    # Install cert-manager
    kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
    
    # Install OPA Gatekeeper
    kubectl apply -f https://raw.githubusercontent.com/open-policy-agent/gatekeeper/release-3.14/deploy/gatekeeper.yaml
    
    # Apply security policies
    kubectl apply -f "${SCRIPT_DIR}/k8s/security-policies.yaml" -n "${NAMESPACE}"
    
    log_success "Azure security setup completed"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check pods
    kubectl get pods -n "${NAMESPACE}"
    
    # Check services
    kubectl get services -n "${NAMESPACE}"
    
    # Check ingress
    kubectl get ingress -n "${NAMESPACE}"
    
    # Check HPA
    kubectl get hpa -n "${NAMESPACE}"
    
    log_success "Deployment verification completed"
}

# Get deployment info
get_deployment_info() {
    log_info "Getting deployment information..."
    
    echo "=== Azure Deployment Information ==="
    echo "Resource Group: ${RESOURCE_GROUP}"
    echo "Region: ${AZURE_REGION}"
    echo "Cluster: ${AKS_CLUSTER_NAME}"
    echo "Namespace: ${NAMESPACE}"
    echo "ACR: ${ACR_NAME}"
    echo "Image: ${ACR_NAME}.azurecr.io/${PROJECT_NAME}:${IMAGE_TAG}"
    echo ""
    
    echo "=== Access Information ==="
    echo "Grafana: http://$(kubectl get ingress -n monitoring -o jsonpath='{.items[0].status.loadBalancer.ingress[0].hostname}')"
    echo "Kibana: http://$(kubectl get ingress -n logging -o jsonpath='{.items[0].status.loadBalancer.ingress[0].hostname}')"
    echo "TruthGPT API: http://$(kubectl get ingress -n "${NAMESPACE}" -o jsonpath='{.items[0].status.loadBalancer.ingress[0].hostname}')"
    echo ""
    
    echo "=== Useful Commands ==="
    echo "kubectl get pods -n ${NAMESPACE}"
    echo "kubectl logs -f deployment/truthgpt-enterprise-optimizer -n ${NAMESPACE}"
    echo "kubectl port-forward svc/truthgpt-enterprise-optimizer 8080:80 -n ${NAMESPACE}"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up Azure resources..."
    
    # Delete application
    kubectl delete -f "${SCRIPT_DIR}/k8s/" -n "${NAMESPACE}" --ignore-not-found=true
    
    # Delete namespace
    kubectl delete namespace "${NAMESPACE}" --ignore-not-found=true
    
    # Delete cluster (optional)
    if [[ "${CLEANUP_CLUSTER:-false}" == "true" ]]; then
        az aks delete --name "${AKS_CLUSTER_NAME}" --resource-group "${RESOURCE_GROUP}" --yes
    fi
    
    log_success "Azure cleanup completed"
}

# Main function
main() {
    log_info "Starting Enterprise TruthGPT Azure Deployment"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --cleanup)
                cleanup
                exit 0
                ;;
            --info)
                get_deployment_info
                exit 0
                ;;
            --help)
                echo "Usage: $0 [--cleanup] [--info] [--help]"
                echo "  --cleanup    Clean up all resources"
                echo "  --info       Show deployment information"
                echo "  --help       Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
        shift
    done
    
    # Execute deployment steps
    check_prerequisites
    create_resource_group
    create_acr
    create_aks_cluster
    configure_kubectl
    create_namespace
    build_and_push_image
    deploy_application
    wait_for_deployment
    setup_monitoring
    setup_logging
    setup_security
    verify_deployment
    get_deployment_info
    
    log_success "Enterprise TruthGPT Azure deployment completed successfully!"
}

# Run main function
main "$@"









