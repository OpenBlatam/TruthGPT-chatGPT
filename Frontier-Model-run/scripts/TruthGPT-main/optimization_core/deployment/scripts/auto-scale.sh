#!/bin/bash
# Ultra Speed TruthGPT Auto-Scaling Script
# Automatically scales resources based on demand and optimization performance

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly NAMESPACE="${NAMESPACE:-truthgpt-optimization}"
readonly CLUSTER_NAME="${CLUSTER_NAME:-truthgpt-cluster}"
readonly AWS_REGION="${AWS_REGION:-us-west-2}"
readonly LOG_FILE="/var/log/truthgpt-autoscale.log"

# Scaling thresholds
readonly CPU_THRESHOLD_HIGH=80
readonly CPU_THRESHOLD_LOW=20
readonly MEMORY_THRESHOLD_HIGH=85
readonly MEMORY_THRESHOLD_LOW=30
readonly OPTIMIZATION_THRESHOLD_HIGH=90
readonly OPTIMIZATION_THRESHOLD_LOW=50

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "${LOG_FILE}"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "${LOG_FILE}"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "${LOG_FILE}"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "${LOG_FILE}"
}

# Error handling
trap 'log_error "Script failed at line $LINENO"' ERR

# Get current resource usage
get_resource_usage() {
    log_info "Getting current resource usage..."
    
    # Get CPU usage
    local cpu_usage
    cpu_usage=$(kubectl top nodes --no-headers | awk '{sum+=$2} END {print sum/NR}')
    
    # Get memory usage
    local memory_usage
    memory_usage=$(kubectl top nodes --no-headers | awk '{sum+=$3} END {print sum/NR}')
    
    # Get pod count
    local pod_count
    pod_count=$(kubectl get pods -n "${NAMESPACE}" --no-headers | wc -l)
    
    # Get HPA status
    local hpa_status
    hpa_status=$(kubectl get hpa -n "${NAMESPACE}" -o jsonpath='{.items[0].status.currentReplicas}')
    
    echo "${cpu_usage} ${memory_usage} ${pod_count} ${hpa_status}"
}

# Scale up resources
scale_up() {
    log_info "Scaling up resources..."
    
    # Scale up HPA
    kubectl patch hpa truthgpt-optimizer-hpa -n "${NAMESPACE}" --type='merge' -p='{"spec":{"maxReplicas":20}}'
    
    # Scale up worker pods
    kubectl scale deployment truthgpt-worker -n "${NAMESPACE}" --replicas=10
    
    # Scale up monitor pods
    kubectl scale deployment truthgpt-monitor -n "${NAMESPACE}" --replicas=3
    
    log_success "Resources scaled up successfully"
}

# Scale down resources
scale_down() {
    log_info "Scaling down resources..."
    
    # Scale down HPA
    kubectl patch hpa truthgpt-optimizer-hpa -n "${NAMESPACE}" --type='merge' -p='{"spec":{"maxReplicas":5}}'
    
    # Scale down worker pods
    kubectl scale deployment truthgpt-worker -n "${NAMESPACE}" --replicas=2
    
    # Scale down monitor pods
    kubectl scale deployment truthgpt-monitor -n "${NAMESPACE}" --replicas=1
    
    log_success "Resources scaled down successfully"
}

# Check optimization performance
check_optimization_performance() {
    log_info "Checking optimization performance..."
    
    # Get optimization metrics
    local optimization_metrics
    optimization_metrics=$(kubectl logs -n "${NAMESPACE}" -l app=truthgpt-optimizer --tail=100 | grep "optimization" | tail -1)
    
    if [ -n "${optimization_metrics}" ]; then
        log_success "Optimization metrics found: ${optimization_metrics}"
    else
        log_warning "No optimization metrics found"
    fi
    
    # Get speed improvement
    local speed_improvement
    speed_improvement=$(kubectl logs -n "${NAMESPACE}" -l app=truthgpt-optimizer --tail=100 | grep "speed_improvement" | tail -1)
    
    if [ -n "${speed_improvement}" ]; then
        log_success "Speed improvement: ${speed_improvement}"
    else
        log_warning "No speed improvement metrics found"
    fi
}

# Main auto-scaling function
main_autoscale() {
    log_info "Starting auto-scaling check..."
    
    # Get current resource usage
    local resource_usage
    resource_usage=$(get_resource_usage)
    
    local cpu_usage memory_usage pod_count hpa_status
    read -r cpu_usage memory_usage pod_count hpa_status <<< "${resource_usage}"
    
    log_info "Current usage - CPU: ${cpu_usage}%, Memory: ${memory_usage}%, Pods: ${pod_count}, HPA: ${hpa_status}"
    
    # Check if scaling is needed
    local scale_needed=false
    local scale_direction=""
    
    # High resource usage - scale up
    if (( $(echo "${cpu_usage} > ${CPU_THRESHOLD_HIGH}" | bc -l) )) || (( $(echo "${memory_usage} > ${MEMORY_THRESHOLD_HIGH}" | bc -l) )); then
        scale_needed=true
        scale_direction="up"
        log_warning "High resource usage detected - scaling up"
    fi
    
    # Low resource usage - scale down
    if (( $(echo "${cpu_usage} < ${CPU_THRESHOLD_LOW}" | bc -l) )) && (( $(echo "${memory_usage} < ${MEMORY_THRESHOLD_LOW}" | bc -l) )); then
        scale_needed=true
        scale_direction="down"
        log_info "Low resource usage detected - scaling down"
    fi
    
    # Perform scaling if needed
    if [ "${scale_needed}" = true ]; then
        if [ "${scale_direction}" = "up" ]; then
            scale_up
        elif [ "${scale_direction}" = "down" ]; then
            scale_down
        fi
    else
        log_success "No scaling needed - resources are optimal"
    fi
    
    # Check optimization performance
    check_optimization_performance
    
    log_success "Auto-scaling check completed"
}

# Continuous auto-scaling
continuous_autoscale() {
    log_info "Starting continuous auto-scaling..."
    
    while true; do
        main_autoscale
        sleep 60  # Check every minute
    done
}

# Parse command line arguments
main() {
    case "${1:-autoscale}" in
        "autoscale")
            main_autoscale
            ;;
        "continuous")
            continuous_autoscale
            ;;
        "scale-up")
            scale_up
            ;;
        "scale-down")
            scale_down
            ;;
        "help")
            echo "Usage: $0 [autoscale|continuous|scale-up|scale-down|help]"
            echo "  autoscale   - Run single auto-scaling check"
            echo "  continuous  - Run continuous auto-scaling"
            echo "  scale-up    - Scale up resources"
            echo "  scale-down  - Scale down resources"
            echo "  help        - Show this help message"
            ;;
        *)
            log_error "Unknown command: $1"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"









