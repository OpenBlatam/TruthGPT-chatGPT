#!/bin/bash
# Ultra Speed TruthGPT Monitoring Script
# Monitors system performance, health, and optimization metrics

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly NAMESPACE="${NAMESPACE:-truthgpt-optimization}"
readonly CLUSTER_NAME="${CLUSTER_NAME:-truthgpt-cluster}"
readonly AWS_REGION="${AWS_REGION:-us-west-2}"
readonly LOG_FILE="/var/log/truthgpt-monitor.log"
readonly ALERT_EMAIL="${ALERT_EMAIL:-admin@truthgpt.com}"
readonly SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"

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

# Check prerequisites
check_prerequisites() {
    log_info "Checking monitoring prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install it first."
        exit 1
    fi
    
    # Check aws CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI not found. Please install it first."
        exit 1
    fi
    
    # Check jq
    if ! command -v jq &> /dev/null; then
        log_error "jq not found. Please install it first."
        exit 1
    fi
    
    # Check curl
    if ! command -v curl &> /dev/null; then
        log_error "curl not found. Please install it first."
        exit 1
    fi
    
    log_success "All monitoring prerequisites met"
}

# Get cluster status
get_cluster_status() {
    log_info "Getting cluster status..."
    
    # Get cluster info
    kubectl cluster-info
    
    # Get node status
    kubectl get nodes -o wide
    
    # Get pod status
    kubectl get pods -n "${NAMESPACE}" -o wide
    
    # Get service status
    kubectl get services -n "${NAMESPACE}" -o wide
    
    # Get ingress status
    kubectl get ingress -n "${NAMESPACE}" -o wide
    
    # Get HPA status
    kubectl get hpa -n "${NAMESPACE}" -o wide
    
    log_success "Cluster status retrieved"
}

# Check pod health
check_pod_health() {
    log_info "Checking pod health..."
    
    local unhealthy_pods=0
    
    # Check if pods are running
    local running_pods
    running_pods=$(kubectl get pods -n "${NAMESPACE}" --field-selector=status.phase=Running --no-headers | wc -l)
    
    local total_pods
    total_pods=$(kubectl get pods -n "${NAMESPACE}" --no-headers | wc -l)
    
    if [ "${running_pods}" -lt "${total_pods}" ]; then
        log_warning "Some pods are not running: ${running_pods}/${total_pods}"
        unhealthy_pods=$((unhealthy_pods + 1))
    else
        log_success "All pods are running: ${running_pods}/${total_pods}"
    fi
    
    # Check for failed pods
    local failed_pods
    failed_pods=$(kubectl get pods -n "${NAMESPACE}" --field-selector=status.phase=Failed --no-headers | wc -l)
    
    if [ "${failed_pods}" -gt 0 ]; then
        log_error "Found ${failed_pods} failed pods"
        unhealthy_pods=$((unhealthy_pods + 1))
    fi
    
    # Check for pending pods
    local pending_pods
    pending_pods=$(kubectl get pods -n "${NAMESPACE}" --field-selector=status.phase=Pending --no-headers | wc -l)
    
    if [ "${pending_pods}" -gt 0 ]; then
        log_warning "Found ${pending_pods} pending pods"
        unhealthy_pods=$((unhealthy_pods + 1))
    fi
    
    return ${unhealthy_pods}
}

# Check resource usage
check_resource_usage() {
    log_info "Checking resource usage..."
    
    # Get node resource usage
    kubectl top nodes
    
    # Get pod resource usage
    kubectl top pods -n "${NAMESPACE}"
    
    # Check memory usage
    local memory_usage
    memory_usage=$(kubectl top nodes --no-headers | awk '{sum+=$3} END {print sum/NR}')
    
    if (( $(echo "${memory_usage} > 80" | bc -l) )); then
        log_warning "High memory usage detected: ${memory_usage}%"
    else
        log_success "Memory usage is normal: ${memory_usage}%"
    fi
    
    # Check CPU usage
    local cpu_usage
    cpu_usage=$(kubectl top nodes --no-headers | awk '{sum+=$2} END {print sum/NR}')
    
    if (( $(echo "${cpu_usage} > 80" | bc -l) )); then
        log_warning "High CPU usage detected: ${cpu_usage}%"
    else
        log_success "CPU usage is normal: ${cpu_usage}%"
    fi
}

# Check application health
check_application_health() {
    log_info "Checking application health..."
    
    # Get service endpoints
    local service_url
    service_url=$(kubectl get ingress -n "${NAMESPACE}" -o jsonpath='{.items[0].status.loadBalancer.ingress[0].hostname}')
    
    if [ -z "${service_url}" ]; then
        log_error "No ingress found"
        return 1
    fi
    
    # Check health endpoint
    local health_status
    health_status=$(curl -s -o /dev/null -w "%{http_code}" "http://${service_url}/health" || echo "000")
    
    if [ "${health_status}" = "200" ]; then
        log_success "Application health check passed"
    else
        log_error "Application health check failed: HTTP ${health_status}"
        return 1
    fi
    
    # Check metrics endpoint
    local metrics_status
    metrics_status=$(curl -s -o /dev/null -w "%{http_code}" "http://${service_url}/metrics" || echo "000")
    
    if [ "${metrics_status}" = "200" ]; then
        log_success "Metrics endpoint is accessible"
    else
        log_warning "Metrics endpoint not accessible: HTTP ${metrics_status}"
    fi
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
    
    # Check speed improvement
    local speed_improvement
    speed_improvement=$(kubectl logs -n "${NAMESPACE}" -l app=truthgpt-optimizer --tail=100 | grep "speed_improvement" | tail -1)
    
    if [ -n "${speed_improvement}" ]; then
        log_success "Speed improvement: ${speed_improvement}"
    else
        log_warning "No speed improvement metrics found"
    fi
}

# Check logs for errors
check_logs_for_errors() {
    log_info "Checking logs for errors..."
    
    local error_count=0
    
    # Check for ERROR level logs
    local error_logs
    error_logs=$(kubectl logs -n "${NAMESPACE}" -l app=truthgpt-optimizer --tail=1000 | grep -i "error" | wc -l)
    
    if [ "${error_logs}" -gt 0 ]; then
        log_warning "Found ${error_logs} error logs"
        error_count=$((error_count + error_logs))
    fi
    
    # Check for WARNING level logs
    local warning_logs
    warning_logs=$(kubectl logs -n "${NAMESPACE}" -l app=truthgpt-optimizer --tail=1000 | grep -i "warning" | wc -l)
    
    if [ "${warning_logs}" -gt 0 ]; then
        log_warning "Found ${warning_logs} warning logs"
        error_count=$((error_count + warning_logs))
    fi
    
    # Check for CRITICAL level logs
    local critical_logs
    critical_logs=$(kubectl logs -n "${NAMESPACE}" -l app=truthgpt-optimizer --tail=1000 | grep -i "critical" | wc -l)
    
    if [ "${critical_logs}" -gt 0 ]; then
        log_error "Found ${critical_logs} critical logs"
        error_count=$((error_count + critical_logs))
    fi
    
    if [ "${error_count}" -eq 0 ]; then
        log_success "No errors found in logs"
    else
        log_warning "Total issues found: ${error_count}"
    fi
    
    return ${error_count}
}

# Send alert
send_alert() {
    local alert_type="$1"
    local message="$2"
    
    log_info "Sending ${alert_type} alert: ${message}"
    
    # Send email alert
    if [ -n "${ALERT_EMAIL}" ]; then
        echo "${message}" | mail -s "TruthGPT ${alert_type} Alert" "${ALERT_EMAIL}"
    fi
    
    # Send Slack alert
    if [ -n "${SLACK_WEBHOOK}" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🚨 TruthGPT ${alert_type} Alert: ${message}\"}" \
            "${SLACK_WEBHOOK}"
    fi
}

# Generate monitoring report
generate_monitoring_report() {
    log_info "Generating monitoring report..."
    
    local report_file="/tmp/truthgpt-monitoring-report-$(date +%Y%m%d-%H%M%S).txt"
    
    {
        echo "=== TruthGPT Monitoring Report ==="
        echo "Generated: $(date)"
        echo "Cluster: ${CLUSTER_NAME}"
        echo "Namespace: ${NAMESPACE}"
        echo "Region: ${AWS_REGION}"
        echo ""
        
        echo "=== Cluster Status ==="
        kubectl get nodes -o wide
        echo ""
        
        echo "=== Pod Status ==="
        kubectl get pods -n "${NAMESPACE}" -o wide
        echo ""
        
        echo "=== Service Status ==="
        kubectl get services -n "${NAMESPACE}" -o wide
        echo ""
        
        echo "=== Ingress Status ==="
        kubectl get ingress -n "${NAMESPACE}" -o wide
        echo ""
        
        echo "=== HPA Status ==="
        kubectl get hpa -n "${NAMESPACE}" -o wide
        echo ""
        
        echo "=== Resource Usage ==="
        kubectl top nodes
        kubectl top pods -n "${NAMESPACE}"
        echo ""
        
        echo "=== Recent Logs ==="
        kubectl logs -n "${NAMESPACE}" -l app=truthgpt-optimizer --tail=50
        echo ""
        
    } > "${report_file}"
    
    log_success "Monitoring report generated: ${report_file}"
    echo "${report_file}"
}

# Main monitoring function
main_monitor() {
    log_info "Starting TruthGPT monitoring..."
    
    local alerts_sent=0
    
    # Check prerequisites
    check_prerequisites
    
    # Get cluster status
    get_cluster_status
    
    # Check pod health
    if ! check_pod_health; then
        send_alert "CRITICAL" "Pod health check failed"
        alerts_sent=$((alerts_sent + 1))
    fi
    
    # Check resource usage
    check_resource_usage
    
    # Check application health
    if ! check_application_health; then
        send_alert "CRITICAL" "Application health check failed"
        alerts_sent=$((alerts_sent + 1))
    fi
    
    # Check optimization performance
    check_optimization_performance
    
    # Check logs for errors
    local error_count
    if ! check_logs_for_errors; then
        error_count=$?
        if [ "${error_count}" -gt 10 ]; then
            send_alert "WARNING" "High error count in logs: ${error_count}"
            alerts_sent=$((alerts_sent + 1))
        fi
    fi
    
    # Generate monitoring report
    local report_file
    report_file=$(generate_monitoring_report)
    
    # Summary
    log_info "Monitoring completed"
    log_info "Alerts sent: ${alerts_sent}"
    log_info "Report file: ${report_file}"
    
    if [ "${alerts_sent}" -eq 0 ]; then
        log_success "All systems are healthy"
    else
        log_warning "Some issues detected, alerts sent"
    fi
}

# Continuous monitoring
continuous_monitor() {
    log_info "Starting continuous monitoring..."
    
    while true; do
        main_monitor
        sleep 300  # Check every 5 minutes
    done
}

# Parse command line arguments
main() {
    case "${1:-monitor}" in
        "monitor")
            main_monitor
            ;;
        "continuous")
            continuous_monitor
            ;;
        "report")
            generate_monitoring_report
            ;;
        "health")
            check_application_health
            ;;
        "resources")
            check_resource_usage
            ;;
        "logs")
            check_logs_for_errors
            ;;
        "help")
            echo "Usage: $0 [monitor|continuous|report|health|resources|logs|help]"
            echo "  monitor     - Run single monitoring check"
            echo "  continuous  - Run continuous monitoring"
            echo "  report      - Generate monitoring report"
            echo "  health      - Check application health"
            echo "  resources   - Check resource usage"
            echo "  logs        - Check logs for errors"
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









