#!/bin/bash
# Ultra Speed TruthGPT AWS Deployment Script
# Deploys TruthGPT optimization system to AWS EKS with auto-scaling

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_NAME="ultra-speed-truthgpt"
readonly AWS_REGION="${AWS_REGION:-us-west-2}"
readonly EKS_CLUSTER_NAME="${EKS_CLUSTER_NAME:-truthgpt-cluster}"
readonly NAMESPACE="${NAMESPACE:-truthgpt-optimization}"
readonly DOCKER_REGISTRY="${DOCKER_REGISTRY:-$(aws sts get-caller-identity --query Account --output text).dkr.ecr.${AWS_REGION}.amazonaws.com}"
readonly IMAGE_TAG="${IMAGE_TAG:-latest}"

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
    log_info "Checking prerequisites..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI not found. Please install it first."
        exit 1
    fi
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install it first."
        exit 1
    fi
    
    # Check eksctl
    if ! command -v eksctl &> /dev/null; then
        log_error "eksctl not found. Please install it first."
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install it first."
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured. Please run 'aws configure'."
        exit 1
    fi
    
    log_success "All prerequisites met"
}

# Create EKS cluster
create_eks_cluster() {
    log_info "Creating EKS cluster: ${EKS_CLUSTER_NAME}"
    
    # Check if cluster already exists
    if eksctl get cluster --name="${EKS_CLUSTER_NAME}" --region="${AWS_REGION}" &> /dev/null; then
        log_warning "Cluster ${EKS_CLUSTER_NAME} already exists"
        return 0
    fi
    
    # Create cluster with optimized configuration
    eksctl create cluster \
        --name="${EKS_CLUSTER_NAME}" \
        --region="${AWS_REGION}" \
        --version=1.28 \
        --nodegroup-name=truthgpt-nodes \
        --node-type=m5.2xlarge \
        --nodes=3 \
        --nodes-min=1 \
        --nodes-max=10 \
        --managed \
        --with-oidc \
        --ssh-access \
        --ssh-public-key="${SSH_PUBLIC_KEY:-~/.ssh/id_rsa.pub}" \
        --node-volume-size=100 \
        --node-volume-type=gp3 \
        --enable-ssm \
        --enable-efa \
        --enable-fargate \
        --fargate-profiles="fargate-profile" \
        --tags="Project=TruthGPT,Environment=Production,Optimization=UltraSpeed"
    
    log_success "EKS cluster created successfully"
}

# Configure kubectl
configure_kubectl() {
    log_info "Configuring kubectl for cluster: ${EKS_CLUSTER_NAME}"
    
    aws eks update-kubeconfig \
        --region="${AWS_REGION}" \
        --name="${EKS_CLUSTER_NAME}"
    
    # Verify cluster access
    if ! kubectl get nodes &> /dev/null; then
        log_error "Failed to access EKS cluster"
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
    
    # Get AWS account ID
    local account_id
    account_id=$(aws sts get-caller-identity --query Account --output text)
    
    # Create ECR repository if it doesn't exist
    aws ecr describe-repositories --repository-names="${PROJECT_NAME}" --region="${AWS_REGION}" &> /dev/null || \
    aws ecr create-repository --repository-name="${PROJECT_NAME}" --region="${AWS_REGION}"
    
    # Login to ECR
    aws ecr get-login-password --region="${AWS_REGION}" | \
    docker login --username AWS --password-stdin "${DOCKER_REGISTRY}"
    
    # Build image
    docker build -t "${PROJECT_NAME}:${IMAGE_TAG}" "${SCRIPT_DIR}/../"
    
    # Tag image
    docker tag "${PROJECT_NAME}:${IMAGE_TAG}" "${DOCKER_REGISTRY}/${PROJECT_NAME}:${IMAGE_TAG}"
    
    # Push image
    docker push "${DOCKER_REGISTRY}/${PROJECT_NAME}:${IMAGE_TAG}"
    
    log_success "Docker image built and pushed successfully"
}

# Deploy application
deploy_application() {
    log_info "Deploying TruthGPT optimization system..."
    
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
    
    kubectl wait --for=condition=available --timeout=300s deployment/truthgpt-optimizer -n "${NAMESPACE}"
    
    log_success "Deployment is ready"
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."
    
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
    
    log_success "Monitoring setup completed"
}

# Setup logging
setup_logging() {
    log_info "Setting up logging..."
    
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
    
    log_success "Logging setup completed"
}

# Setup security
setup_security() {
    log_info "Setting up security..."
    
    # Install cert-manager
    kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
    
    # Install OPA Gatekeeper
    kubectl apply -f https://raw.githubusercontent.com/open-policy-agent/gatekeeper/release-3.14/deploy/gatekeeper.yaml
    
    # Apply security policies
    kubectl apply -f "${SCRIPT_DIR}/k8s/security-policies.yaml" -n "${NAMESPACE}"
    
    log_success "Security setup completed"
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
    
    echo "=== Deployment Information ==="
    echo "Cluster: ${EKS_CLUSTER_NAME}"
    echo "Region: ${AWS_REGION}"
    echo "Namespace: ${NAMESPACE}"
    echo "Image: ${DOCKER_REGISTRY}/${PROJECT_NAME}:${IMAGE_TAG}"
    echo ""
    
    echo "=== Access Information ==="
    echo "Grafana: http://$(kubectl get ingress -n monitoring -o jsonpath='{.items[0].status.loadBalancer.ingress[0].hostname}')"
    echo "Kibana: http://$(kubectl get ingress -n logging -o jsonpath='{.items[0].status.loadBalancer.ingress[0].hostname}')"
    echo "TruthGPT API: http://$(kubectl get ingress -n "${NAMESPACE}" -o jsonpath='{.items[0].status.loadBalancer.ingress[0].hostname}')"
    echo ""
    
    echo "=== Useful Commands ==="
    echo "kubectl get pods -n ${NAMESPACE}"
    echo "kubectl logs -f deployment/truthgpt-optimizer -n ${NAMESPACE}"
    echo "kubectl port-forward svc/truthgpt-optimizer 8080:80 -n ${NAMESPACE}"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up resources..."
    
    # Delete application
    kubectl delete -f "${SCRIPT_DIR}/k8s/" -n "${NAMESPACE}" --ignore-not-found=true
    
    # Delete namespace
    kubectl delete namespace "${NAMESPACE}" --ignore-not-found=true
    
    # Delete cluster (optional)
    if [[ "${CLEANUP_CLUSTER:-false}" == "true" ]]; then
        eksctl delete cluster --name="${EKS_CLUSTER_NAME}" --region="${AWS_REGION}"
    fi
    
    log_success "Cleanup completed"
}

# Main function
main() {
    log_info "Starting Ultra Speed TruthGPT AWS Deployment"
    
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
    create_eks_cluster
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
    
    log_success "Ultra Speed TruthGPT deployment completed successfully!"
}

# Run main function
main "$@"









