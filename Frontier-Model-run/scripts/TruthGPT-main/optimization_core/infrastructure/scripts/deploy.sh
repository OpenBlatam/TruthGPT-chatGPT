#!/bin/bash
# TruthGPT Deployment Script
# ==========================
# Comprehensive deployment script for TruthGPT optimization framework

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
NAMESPACE="${NAMESPACE:-truthgpt-optimization}"
ENVIRONMENT="${ENVIRONMENT:-production}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REPLICA_COUNT="${REPLICA_COUNT:-3}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
trap 'log_error "Deployment failed at line $LINENO"' ERR

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check if helm is installed
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed"
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed"
        exit 1
    fi
    
    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create namespace
create_namespace() {
    log_info "Creating namespace: $NAMESPACE"
    
    kubectl apply -f - <<EOF
apiVersion: v1
kind: Namespace
metadata:
  name: $NAMESPACE
  labels:
    name: $NAMESPACE
    environment: $ENVIRONMENT
    team: ml-engineering
    cost-center: ai-research
EOF
    
    log_success "Namespace created successfully"
}

# Deploy secrets
deploy_secrets() {
    log_info "Deploying secrets..."
    
    # Check if secrets exist
    if kubectl get secret truthgpt-secrets -n "$NAMESPACE" &> /dev/null; then
        log_warning "Secrets already exist, skipping..."
        return
    fi
    
    # Create secrets from environment variables
    kubectl create secret generic truthgpt-secrets \
        --from-literal=secret-key="${SECRET_KEY:-$(openssl rand -base64 32)}" \
        --from-literal=jwt-secret="${JWT_SECRET:-$(openssl rand -base64 32)}" \
        --from-literal=redis-password="${REDIS_PASSWORD:-$(openssl rand -base64 32)}" \
        --from-literal=wandb-api-key="${WANDB_API_KEY:-}" \
        --from-literal=azure-storage-key="${AZURE_STORAGE_KEY:-}" \
        -n "$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    log_success "Secrets deployed successfully"
}

# Deploy configmaps
deploy_configmaps() {
    log_info "Deploying configmaps..."
    
    kubectl apply -f "$SCRIPT_DIR/../kubernetes/configmap.yaml"
    
    log_success "Configmaps deployed successfully"
}

# Deploy persistent volume claims
deploy_pvcs() {
    log_info "Deploying persistent volume claims..."
    
    kubectl apply -f "$SCRIPT_DIR/../kubernetes/pvc.yaml"
    
    # Wait for PVCs to be bound
    log_info "Waiting for PVCs to be bound..."
    kubectl wait --for=condition=Bound pvc/truthgpt-logs-pvc -n "$NAMESPACE" --timeout=300s
    kubectl wait --for=condition=Bound pvc/truthgpt-cache-pvc -n "$NAMESPACE" --timeout=300s
    kubectl wait --for=condition=Bound pvc/truthgpt-models-pvc -n "$NAMESPACE" --timeout=300s
    
    log_success "PVCs deployed successfully"
}

# Deploy RBAC
deploy_rbac() {
    log_info "Deploying RBAC configuration..."
    
    kubectl apply -f "$SCRIPT_DIR/../kubernetes/rbac.yaml"
    
    log_success "RBAC deployed successfully"
}

# Deploy application
deploy_application() {
    log_info "Deploying TruthGPT application..."
    
    # Update image tag in deployment
    sed "s|image: truthgpt/optimization:latest|image: truthgpt/optimization:$IMAGE_TAG|g" \
        "$SCRIPT_DIR/../kubernetes/deployment.yaml" | \
    sed "s|replicas: 3|replicas: $REPLICA_COUNT|g" | \
    kubectl apply -f -
    
    log_success "Application deployed successfully"
}

# Deploy services
deploy_services() {
    log_info "Deploying services..."
    
    kubectl apply -f "$SCRIPT_DIR/../kubernetes/service.yaml"
    
    log_success "Services deployed successfully"
}

# Deploy HPA
deploy_hpa() {
    log_info "Deploying Horizontal Pod Autoscaler..."
    
    kubectl apply -f "$SCRIPT_DIR/../kubernetes/hpa.yaml"
    
    log_success "HPA deployed successfully"
}

# Deploy network policies
deploy_network_policies() {
    log_info "Deploying network policies..."
    
    kubectl apply -f "$SCRIPT_DIR/../kubernetes/network-policy.yaml"
    
    log_success "Network policies deployed successfully"
}

# Wait for deployment
wait_for_deployment() {
    log_info "Waiting for deployment to be ready..."
    
    kubectl wait --for=condition=available --timeout=600s \
        deployment/truthgpt-optimization -n "$NAMESPACE"
    
    log_success "Deployment is ready!"
}

# Health check
health_check() {
    log_info "Running health checks..."
    
    # Check if pods are running
    local running_pods=$(kubectl get pods -n "$NAMESPACE" -l app=truthgpt-optimization --field-selector=status.phase=Running --no-headers | wc -l)
    
    if [ "$running_pods" -eq 0 ]; then
        log_error "No running pods found"
        exit 1
    fi
    
    log_success "Health checks passed ($running_pods pods running)"
}

# Get service information
get_service_info() {
    log_info "Getting service information..."
    
    # Get external IP
    local external_ip=$(kubectl get service truthgpt-api-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "Pending")
    
    log_success "Service information:"
    echo "  Namespace: $NAMESPACE"
    echo "  External IP: $external_ip"
    echo "  Port: 80 (HTTP), 443 (HTTPS), 9090 (Metrics)"
    
    if [ "$external_ip" != "Pending" ] && [ "$external_ip" != "" ]; then
        echo "  Health check: http://$external_ip/health"
        echo "  API documentation: http://$external_ip/docs"
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    # Add cleanup logic here if needed
}

# Main deployment function
deploy() {
    log_info "Starting TruthGPT deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Namespace: $NAMESPACE"
    log_info "Image tag: $IMAGE_TAG"
    log_info "Replica count: $REPLICA_COUNT"
    
    check_prerequisites
    create_namespace
    deploy_secrets
    deploy_configmaps
    deploy_pvcs
    deploy_rbac
    deploy_application
    deploy_services
    deploy_hpa
    deploy_network_policies
    wait_for_deployment
    health_check
    get_service_info
    
    log_success "TruthGPT deployment completed successfully!"
}

# Rollback function
rollback() {
    log_info "Rolling back deployment..."
    
    kubectl delete -f "$SCRIPT_DIR/../kubernetes/" --ignore-not-found=true
    
    log_success "Rollback completed"
}

# Show usage
usage() {
    echo "Usage: $0 [OPTIONS] COMMAND"
    echo ""
    echo "Commands:"
    echo "  deploy     Deploy TruthGPT application"
    echo "  rollback   Rollback deployment"
    echo "  status     Show deployment status"
    echo "  logs       Show application logs"
    echo ""
    echo "Options:"
    echo "  -n, --namespace NAMESPACE    Kubernetes namespace (default: truthgpt-optimization)"
    echo "  -e, --environment ENV        Environment (default: production)"
    echo "  -t, --tag TAG               Image tag (default: latest)"
    echo "  -r, --replicas COUNT        Replica count (default: 3)"
    echo "  -h, --help                 Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  SECRET_KEY                  Application secret key"
    echo "  JWT_SECRET                  JWT secret key"
    echo "  REDIS_PASSWORD              Redis password"
    echo "  WANDB_API_KEY               Weights & Biases API key"
    echo "  AZURE_STORAGE_KEY           Azure storage key"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -r|--replicas)
            REPLICA_COUNT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        deploy)
            COMMAND="deploy"
            shift
            ;;
        rollback)
            COMMAND="rollback"
            shift
            ;;
        status)
            COMMAND="status"
            shift
            ;;
        logs)
            COMMAND="logs"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Execute command
case ${COMMAND:-deploy} in
    deploy)
        deploy
        ;;
    rollback)
        rollback
        ;;
    status)
        kubectl get all -n "$NAMESPACE"
        ;;
    logs)
        kubectl logs -l app=truthgpt-optimization -n "$NAMESPACE" --tail=100 -f
        ;;
    *)
        log_error "Unknown command: ${COMMAND:-deploy}"
        usage
        exit 1
        ;;
esac


