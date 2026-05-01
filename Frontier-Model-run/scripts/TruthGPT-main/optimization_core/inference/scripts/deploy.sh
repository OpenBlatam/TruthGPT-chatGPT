#!/bin/bash
# 🚀 Deployment Script
# Automated deployment script for inference API

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${ENVIRONMENT:-production}
NAMESPACE=${NAMESPACE:-inference}
IMAGE_TAG=${IMAGE_TAG:-latest}
REGISTRY=${REGISTRY:-ghcr.io}

echo -e "${GREEN}🚀 Starting deployment...${NC}"

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    command -v kubectl >/dev/null 2>&1 || { echo -e "${RED}kubectl is required${NC}"; exit 1; }
    command -v docker >/dev/null 2>&1 || { echo -e "${RED}docker is required${NC}"; exit 1; }
    
    kubectl cluster-info >/dev/null 2>&1 || { echo -e "${RED}Not connected to Kubernetes cluster${NC}"; exit 1; }
    
    echo -e "${GREEN}✓ Prerequisites OK${NC}"
}

# Build Docker image
build_image() {
    echo -e "${YELLOW}Building Docker image...${NC}"
    docker build -t ${REGISTRY}/frontier-inference-api:${IMAGE_TAG} -f Dockerfile ..
    echo -e "${GREEN}✓ Image built${NC}"
}

# Push image
push_image() {
    if [ "$SKIP_PUSH" != "true" ]; then
        echo -e "${YELLOW}Pushing image...${NC}"
        docker push ${REGISTRY}/frontier-inference-api:${IMAGE_TAG}
        echo -e "${GREEN}✓ Image pushed${NC}"
    fi
}

# Create namespace
create_namespace() {
    echo -e "${YELLOW}Creating namespace...${NC}"
    kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
    echo -e "${GREEN}✓ Namespace ready${NC}"
}

# Apply secrets
apply_secrets() {
    if [ -f "k8s/secrets.yaml" ]; then
        echo -e "${YELLOW}Applying secrets...${NC}"
        kubectl apply -f k8s/secrets.yaml -n ${NAMESPACE}
        echo -e "${GREEN}✓ Secrets applied${NC}"
    else
        echo -e "${YELLOW}⚠ No secrets file found, skipping...${NC}"
    fi
}

# Update deployment
update_deployment() {
    echo -e "${YELLOW}Updating deployment...${NC}"
    
    # Update image in deployment
    sed "s|image:.*|image: ${REGISTRY}/frontier-inference-api:${IMAGE_TAG}|g" k8s/deployment.yaml | \
        kubectl apply -f - -n ${NAMESPACE}
    
    echo -e "${GREEN}✓ Deployment updated${NC}"
}

# Wait for rollout
wait_for_rollout() {
    echo -e "${YELLOW}Waiting for rollout...${NC}"
    kubectl rollout status deployment/inference-api -n ${NAMESPACE} --timeout=10m
    echo -e "${GREEN}✓ Rollout complete${NC}"
}

# Health check
health_check() {
    echo -e "${YELLOW}Running health check...${NC}"
    
    # Get service endpoint
    SERVICE_URL=$(kubectl get svc inference-api -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
    
    if [ -z "$SERVICE_URL" ]; then
        SERVICE_URL="http://localhost:8080"  # Port-forward fallback
    fi
    
    # Try health endpoint
    for i in {1..10}; do
        if curl -f -s ${SERVICE_URL}/health > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Health check passed${NC}"
            return 0
        fi
        echo -e "${YELLOW}Waiting for service... (attempt $i/10)${NC}"
        sleep 5
    done
    
    echo -e "${RED}✗ Health check failed${NC}"
    return 1
}

# Main deployment
main() {
    check_prerequisites
    
    if [ "$SKIP_BUILD" != "true" ]; then
        build_image
        push_image
    fi
    
    create_namespace
    apply_secrets
    update_deployment
    wait_for_rollout
    
    if [ "$SKIP_HEALTH_CHECK" != "true" ]; then
        health_check
    fi
    
    echo -e "${GREEN}🎉 Deployment complete!${NC}"
    
    # Show status
    echo -e "\n${YELLOW}Deployment Status:${NC}"
    kubectl get pods -n ${NAMESPACE}
    kubectl get svc -n ${NAMESPACE}
}

# Handle flags
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-push)
            SKIP_PUSH=true
            shift
            ;;
        --skip-health-check)
            SKIP_HEALTH_CHECK=true
            shift
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

main


