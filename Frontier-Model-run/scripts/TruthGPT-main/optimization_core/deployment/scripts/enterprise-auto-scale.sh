#!/bin/bash
# Enterprise TruthGPT Auto-Scaling Script
# Advanced auto-scaling with enterprise features

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly NAMESPACE="${NAMESPACE:-truthgpt-enterprise}"
readonly CLUSTER_NAME="${CLUSTER_NAME:-truthgpt-enterprise-cluster}"
readonly AZURE_REGION="${AZURE_REGION:-East US}"
readonly RESOURCE_GROUP="${RESOURCE_GROUP:-truthgpt-enterprise-rg}"
readonly LOG_FILE="/var/log/truthgpt-enterprise-autoscale.log"

# Enterprise scaling thresholds
readonly CPU_THRESHOLD_HIGH=80
readonly CPU_THRESHOLD_LOW=20
readonly MEMORY_THRESHOLD_HIGH=85
readonly MEMORY_THRESHOLD_LOW=30
readonly DISK_THRESHOLD_HIGH=90
readonly DISK_THRESHOLD_LOW=20
readonly NETWORK_THRESHOLD_HIGH=80
readonly NETWORK_THRESHOLD_LOW=20
readonly OPTIMIZATION_THRESHOLD_HIGH=90
readonly OPTIMIZATION_THRESHOLD_LOW=50
readonly SECURITY_THRESHOLD_HIGH=95
readonly SECURITY_THRESHOLD_LOW=80
readonly COMPLIANCE_THRESHOLD_HIGH=95
readonly COMPLIANCE_THRESHOLD_LOW=80
readonly COST_THRESHOLD_HIGH=90
readonly COST_THRESHOLD_LOW=50

# Enterprise scaling limits
readonly MIN_REPLICAS=1
readonly MAX_REPLICAS=50
readonly MIN_NODES=1
readonly MAX_NODES=20
readonly MIN_CPU=1000m
readonly MAX_CPU=8000m
readonly MIN_MEMORY=2Gi
readonly MAX_MEMORY=16Gi

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
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

log_enterprise() {
    echo -e "${PURPLE}[ENTERPRISE]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "${LOG_FILE}"
}

log_scaling() {
    echo -e "${CYAN}[SCALING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "${LOG_FILE}"
}

# Error handling
trap 'log_error "Script failed at line $LINENO"' ERR

# Get current resource usage
get_resource_usage() {
    log_info "Getting enterprise resource usage..."
    
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
    
    # Get VPA status
    local vpa_status
    vpa_status=$(kubectl get vpa -n "${NAMESPACE}" -o jsonpath='{.items[0].status.currentReplicas}')
    
    # Get node count
    local node_count
    node_count=$(kubectl get nodes --no-headers | wc -l)
    
    # Get disk usage
    local disk_usage
    disk_usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
    
    # Get network usage
    local network_usage
    network_usage=$(cat /proc/net/dev | grep -v "lo:" | awk '{sum+=$2+$10} END {print sum/1024/1024}')
    
    echo "${cpu_usage} ${memory_usage} ${pod_count} ${hpa_status} ${vpa_status} ${node_count} ${disk_usage} ${network_usage}"
}

# Scale up resources
scale_up_resources() {
    log_scaling "Scaling up enterprise resources..."
    
    # Scale up HPA
    kubectl patch hpa truthgpt-enterprise-optimizer-hpa -n "${NAMESPACE}" --type='merge' -p='{"spec":{"maxReplicas":'${MAX_REPLICAS}'}}'
    
    # Scale up worker pods
    kubectl scale deployment truthgpt-enterprise-worker -n "${NAMESPACE}" --replicas=20
    
    # Scale up monitor pods
    kubectl scale deployment truthgpt-enterprise-monitor -n "${NAMESPACE}" --replicas=5
    
    # Scale up VPA
    kubectl patch vpa truthgpt-enterprise-optimizer-vpa -n "${NAMESPACE}" --type='merge' -p='{"spec":{"updatePolicy":{"updateMode":"Auto"}}}'
    
    # Scale up nodes (if using cluster autoscaler)
    kubectl patch node --type='merge' -p='{"metadata":{"annotations":{"cluster-autoscaler.kubernetes.io/scale-down-disabled":"true"}}}'
    
    log_success "Enterprise resources scaled up successfully"
}

# Scale down resources
scale_down_resources() {
    log_scaling "Scaling down enterprise resources..."
    
    # Scale down HPA
    kubectl patch hpa truthgpt-enterprise-optimizer-hpa -n "${NAMESPACE}" --type='merge' -p='{"spec":{"maxReplicas":'${MIN_REPLICAS}'}}'
    
    # Scale down worker pods
    kubectl scale deployment truthgpt-enterprise-worker -n "${NAMESPACE}" --replicas=2
    
    # Scale down monitor pods
    kubectl scale deployment truthgpt-enterprise-monitor -n "${NAMESPACE}" --replicas=1
    
    # Scale down VPA
    kubectl patch vpa truthgpt-enterprise-optimizer-vpa -n "${NAMESPACE}" --type='merge' -p='{"spec":{"updatePolicy":{"updateMode":"Off"}}}'
    
    # Scale down nodes (if using cluster autoscaler)
    kubectl patch node --type='merge' -p='{"metadata":{"annotations":{"cluster-autoscaler.kubernetes.io/scale-down-disabled":"false"}}}'
    
    log_success "Enterprise resources scaled down successfully"
}

# Scale based on optimization performance
scale_based_on_optimization() {
    log_enterprise "Scaling based on optimization performance..."
    
    # Get optimization metrics
    local optimization_metrics
    optimization_metrics=$(kubectl logs -n "${NAMESPACE}" -l app=truthgpt-enterprise-optimizer --tail=100 | grep "optimization" | tail -1)
    
    if [ -n "${optimization_metrics}" ]; then
        log_enterprise "Optimization metrics found: ${optimization_metrics}"
        
        # Extract optimization score
        local optimization_score
        optimization_score=$(echo "${optimization_metrics}" | grep -o '[0-9]*\.[0-9]*' | head -1)
        
        if [ -n "${optimization_score}" ]; then
            if (( $(echo "${optimization_score} > ${OPTIMIZATION_THRESHOLD_HIGH}" | bc -l) )); then
                log_enterprise "High optimization performance detected: ${optimization_score}%"
                scale_up_resources
            elif (( $(echo "${optimization_score} < ${OPTIMIZATION_THRESHOLD_LOW}" | bc -l) )); then
                log_enterprise "Low optimization performance detected: ${optimization_score}%"
                scale_down_resources
            else
                log_enterprise "Optimization performance is normal: ${optimization_score}%"
            fi
        fi
    else
        log_warning "No optimization metrics found"
    fi
}

# Scale based on security metrics
scale_based_on_security() {
    log_enterprise "Scaling based on security metrics..."
    
    # Get security metrics
    local security_metrics
    security_metrics=$(kubectl logs -n "${NAMESPACE}" -l app=truthgpt-enterprise-optimizer --tail=100 | grep "security" | tail -1)
    
    if [ -n "${security_metrics}" ]; then
        log_enterprise "Security metrics found: ${security_metrics}"
        
        # Extract security score
        local security_score
        security_score=$(echo "${security_metrics}" | grep -o '[0-9]*\.[0-9]*' | head -1)
        
        if [ -n "${security_score}" ]; then
            if (( $(echo "${security_score} > ${SECURITY_THRESHOLD_HIGH}" | bc -l) )); then
                log_enterprise "High security score detected: ${security_score}%"
                scale_up_resources
            elif (( $(echo "${security_score} < ${SECURITY_THRESHOLD_LOW}" | bc -l) )); then
                log_enterprise "Low security score detected: ${security_score}%"
                scale_down_resources
            else
                log_enterprise "Security score is normal: ${security_score}%"
            fi
        fi
    else
        log_warning "No security metrics found"
    fi
}

# Scale based on compliance metrics
scale_based_on_compliance() {
    log_enterprise "Scaling based on compliance metrics..."
    
    # Get compliance metrics
    local compliance_metrics
    compliance_metrics=$(kubectl logs -n "${NAMESPACE}" -l app=truthgpt-enterprise-optimizer --tail=100 | grep "compliance" | tail -1)
    
    if [ -n "${compliance_metrics}" ]; then
        log_enterprise "Compliance metrics found: ${compliance_metrics}"
        
        # Extract compliance score
        local compliance_score
        compliance_score=$(echo "${compliance_metrics}" | grep -o '[0-9]*\.[0-9]*' | head -1)
        
        if [ -n "${compliance_score}" ]; then
            if (( $(echo "${compliance_score} > ${COMPLIANCE_THRESHOLD_HIGH}" | bc -l) )); then
                log_enterprise "High compliance score detected: ${compliance_score}%"
                scale_up_resources
            elif (( $(echo "${compliance_score} < ${COMPLIANCE_THRESHOLD_LOW}" | bc -l) )); then
                log_enterprise "Low compliance score detected: ${compliance_score}%"
                scale_down_resources
            else
                log_enterprise "Compliance score is normal: ${compliance_score}%"
            fi
        fi
    else
        log_warning "No compliance metrics found"
    fi
}

# Scale based on cost optimization
scale_based_on_cost() {
    log_enterprise "Scaling based on cost optimization..."
    
    # Get cost optimization metrics
    local cost_metrics
    cost_metrics=$(kubectl logs -n "${NAMESPACE}" -l app=truthgpt-enterprise-optimizer --tail=100 | grep "cost" | tail -1)
    
    if [ -n "${cost_metrics}" ]; then
        log_enterprise "Cost optimization metrics found: ${cost_metrics}"
        
        # Extract cost score
        local cost_score
        cost_score=$(echo "${cost_metrics}" | grep -o '[0-9]*\.[0-9]*' | head -1)
        
        if [ -n "${cost_score}" ]; then
            if (( $(echo "${cost_score} > ${COST_THRESHOLD_HIGH}" | bc -l) )); then
                log_enterprise "High cost optimization detected: ${cost_score}%"
                scale_up_resources
            elif (( $(echo "${cost_score} < ${COST_THRESHOLD_LOW}" | bc -l) )); then
                log_enterprise "Low cost optimization detected: ${cost_score}%"
                scale_down_resources
            else
                log_enterprise "Cost optimization is normal: ${cost_score}%"
            fi
        fi
    else
        log_warning "No cost optimization metrics found"
    fi
}

# Check enterprise scaling conditions
check_enterprise_scaling_conditions() {
    log_info "Checking enterprise scaling conditions..."
    
    # Get current resource usage
    local resource_usage
    resource_usage=$(get_resource_usage)
    
    local cpu_usage memory_usage pod_count hpa_status vpa_status node_count disk_usage network_usage
    read -r cpu_usage memory_usage pod_count hpa_status vpa_status node_count disk_usage network_usage <<< "${resource_usage}"
    
    log_info "Current usage - CPU: ${cpu_usage}%, Memory: ${memory_usage}%, Pods: ${pod_count}, HPA: ${hpa_status}, VPA: ${vpa_status}, Nodes: ${node_count}, Disk: ${disk_usage}%, Network: ${network_usage}MB"
    
    # Check if scaling is needed
    local scale_needed=false
    local scale_direction=""
    local scale_reason=""
    
    # High resource usage - scale up
    if (( $(echo "${cpu_usage} > ${CPU_THRESHOLD_HIGH}" | bc -l) )) || (( $(echo "${memory_usage} > ${MEMORY_THRESHOLD_HIGH}" | bc -l) )); then
        scale_needed=true
        scale_direction="up"
        scale_reason="High resource usage"
        log_warning "High resource usage detected - scaling up"
    fi
    
    # Low resource usage - scale down
    if (( $(echo "${cpu_usage} < ${CPU_THRESHOLD_LOW}" | bc -l) )) && (( $(echo "${memory_usage} < ${MEMORY_THRESHOLD_LOW}" | bc -l) )); then
        scale_needed=true
        scale_direction="down"
        scale_reason="Low resource usage"
        log_info "Low resource usage detected - scaling down"
    fi
    
    # High disk usage - scale up
    if [ "${disk_usage}" -gt "${DISK_THRESHOLD_HIGH}" ]; then
        scale_needed=true
        scale_direction="up"
        scale_reason="High disk usage"
        log_warning "High disk usage detected - scaling up"
    fi
    
    # Low disk usage - scale down
    if [ "${disk_usage}" -lt "${DISK_THRESHOLD_LOW}" ]; then
        scale_needed=true
        scale_direction="down"
        scale_reason="Low disk usage"
        log_info "Low disk usage detected - scaling down"
    fi
    
    # High network usage - scale up
    if (( $(echo "${network_usage} > ${NETWORK_THRESHOLD_HIGH}" | bc -l) )); then
        scale_needed=true
        scale_direction="up"
        scale_reason="High network usage"
        log_warning "High network usage detected - scaling up"
    fi
    
    # Low network usage - scale down
    if (( $(echo "${network_usage} < ${NETWORK_THRESHOLD_LOW}" | bc -l) )); then
        scale_needed=true
        scale_direction="down"
        scale_reason="Low network usage"
        log_info "Low network usage detected - scaling down"
    fi
    
    # Perform scaling if needed
    if [ "${scale_needed}" = true ]; then
        log_scaling "Scaling ${scale_direction} due to: ${scale_reason}"
        if [ "${scale_direction}" = "up" ]; then
            scale_up_resources
        elif [ "${scale_direction}" = "down" ]; then
            scale_down_resources
        fi
    else
        log_success "No scaling needed - resources are optimal"
    fi
    
    # Check enterprise-specific scaling conditions
    scale_based_on_optimization
    scale_based_on_security
    scale_based_on_compliance
    scale_based_on_cost
    
    log_success "Enterprise scaling check completed"
}

# Main enterprise auto-scaling function
main_enterprise_autoscale() {
    log_info "Starting enterprise auto-scaling check..."
    
    # Check enterprise scaling conditions
    check_enterprise_scaling_conditions
    
    log_success "Enterprise auto-scaling check completed"
}

# Continuous enterprise auto-scaling
continuous_enterprise_autoscale() {
    log_info "Starting continuous enterprise auto-scaling..."
    
    while true; do
        main_enterprise_autoscale
        sleep 60  # Check every minute
    done
}

# Parse command line arguments
main() {
    case "${1:-autoscale}" in
        "autoscale")
            main_enterprise_autoscale
            ;;
        "continuous")
            continuous_enterprise_autoscale
            ;;
        "scale-up")
            scale_up_resources
            ;;
        "scale-down")
            scale_down_resources
            ;;
        "optimization")
            scale_based_on_optimization
            ;;
        "security")
            scale_based_on_security
            ;;
        "compliance")
            scale_based_on_compliance
            ;;
        "cost")
            scale_based_on_cost
            ;;
        "help")
            echo "Usage: $0 [autoscale|continuous|scale-up|scale-down|optimization|security|compliance|cost|help]"
            echo "  autoscale   - Run single enterprise auto-scaling check"
            echo "  continuous  - Run continuous enterprise auto-scaling"
            echo "  scale-up    - Scale up enterprise resources"
            echo "  scale-down  - Scale down enterprise resources"
            echo "  optimization - Scale based on optimization performance"
            echo "  security    - Scale based on security metrics"
            echo "  compliance  - Scale based on compliance metrics"
            echo "  cost        - Scale based on cost optimization"
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









