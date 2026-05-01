#!/bin/bash
# Enterprise TruthGPT Monitoring Script
# Advanced monitoring with enterprise features

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly NAMESPACE="${NAMESPACE:-truthgpt-enterprise}"
readonly CLUSTER_NAME="${CLUSTER_NAME:-truthgpt-enterprise-cluster}"
readonly AZURE_REGION="${AZURE_REGION:-East US}"
readonly RESOURCE_GROUP="${RESOURCE_GROUP:-truthgpt-enterprise-rg}"
readonly LOG_FILE="/var/log/truthgpt-enterprise-monitor.log"
readonly ALERT_EMAIL="${ALERT_EMAIL:-admin@truthgpt.com}"
readonly SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"
readonly TEAMS_WEBHOOK="${TEAMS_WEBHOOK:-}"

# Enterprise monitoring thresholds
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

log_critical() {
    echo -e "${RED}[CRITICAL]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "${LOG_FILE}"
}

log_enterprise() {
    echo -e "${PURPLE}[ENTERPRISE]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "${LOG_FILE}"
}

log_security() {
    echo -e "${CYAN}[SECURITY]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "${LOG_FILE}"
}

# Error handling
trap 'log_error "Script failed at line $LINENO"' ERR

# Check prerequisites
check_prerequisites() {
    log_info "Checking enterprise monitoring prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install it first."
        exit 1
    fi
    
    # Check az CLI
    if ! command -v az &> /dev/null; then
        log_error "Azure CLI not found. Please install it first."
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
    
    # Check Azure login
    if ! az account show &> /dev/null; then
        log_error "Not logged in to Azure. Please run 'az login'."
        exit 1
    fi
    
    # Check cluster access
    if ! kubectl get nodes &> /dev/null; then
        log_error "Cannot access Kubernetes cluster"
        exit 1
    fi
    
    log_success "All enterprise monitoring prerequisites met"
}

# Get cluster status
get_cluster_status() {
    log_info "Getting enterprise cluster status..."
    
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
    
    # Get VPA status
    kubectl get vpa -n "${NAMESPACE}" -o wide
    
    # Get PDB status
    kubectl get pdb -n "${NAMESPACE}" -o wide
    
    log_success "Enterprise cluster status retrieved"
}

# Check pod health
check_pod_health() {
    log_info "Checking enterprise pod health..."
    
    local unhealthy_pods=0
    local critical_pods=0
    
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
        log_critical "Found ${failed_pods} failed pods"
        critical_pods=$((critical_pods + 1))
    fi
    
    # Check for pending pods
    local pending_pods
    pending_pods=$(kubectl get pods -n "${NAMESPACE}" --field-selector=status.phase=Pending --no-headers | wc -l)
    
    if [ "${pending_pods}" -gt 0 ]; then
        log_warning "Found ${pending_pods} pending pods"
        unhealthy_pods=$((unhealthy_pods + 1))
    fi
    
    # Check for crash loop backoff pods
    local crash_loop_pods
    crash_loop_pods=$(kubectl get pods -n "${NAMESPACE}" --no-headers | grep -c "CrashLoopBackOff" || true)
    
    if [ "${crash_loop_pods}" -gt 0 ]; then
        log_critical "Found ${crash_loop_pods} pods in CrashLoopBackOff"
        critical_pods=$((critical_pods + 1))
    fi
    
    return $((unhealthy_pods + critical_pods))
}

# Check resource usage
check_resource_usage() {
    log_info "Checking enterprise resource usage..."
    
    # Get node resource usage
    kubectl top nodes
    
    # Get pod resource usage
    kubectl top pods -n "${NAMESPACE}"
    
    # Check memory usage
    local memory_usage
    memory_usage=$(kubectl top nodes --no-headers | awk '{sum+=$3} END {print sum/NR}')
    
    if (( $(echo "${memory_usage} > ${MEMORY_THRESHOLD_HIGH}" | bc -l) )); then
        log_warning "High memory usage detected: ${memory_usage}%"
    else
        log_success "Memory usage is normal: ${memory_usage}%"
    fi
    
    # Check CPU usage
    local cpu_usage
    cpu_usage=$(kubectl top nodes --no-headers | awk '{sum+=$2} END {print sum/NR}')
    
    if (( $(echo "${cpu_usage} > ${CPU_THRESHOLD_HIGH}" | bc -l) )); then
        log_warning "High CPU usage detected: ${cpu_usage}%"
    else
        log_success "CPU usage is normal: ${cpu_usage}%"
    fi
    
    # Check disk usage
    local disk_usage
    disk_usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
    
    if [ "${disk_usage}" -gt "${DISK_THRESHOLD_HIGH}" ]; then
        log_warning "High disk usage detected: ${disk_usage}%"
    else
        log_success "Disk usage is normal: ${disk_usage}%"
    fi
}

# Check application health
check_application_health() {
    log_info "Checking enterprise application health..."
    
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
    
    # Check readiness endpoint
    local readiness_status
    readiness_status=$(curl -s -o /dev/null -w "%{http_code}" "http://${service_url}/ready" || echo "000")
    
    if [ "${readiness_status}" = "200" ]; then
        log_success "Readiness check passed"
    else
        log_warning "Readiness check failed: HTTP ${readiness_status}"
    fi
}

# Check optimization performance
check_optimization_performance() {
    log_info "Checking enterprise optimization performance..."
    
    # Get optimization metrics
    local optimization_metrics
    optimization_metrics=$(kubectl logs -n "${NAMESPACE}" -l app=truthgpt-enterprise-optimizer --tail=100 | grep "optimization" | tail -1)
    
    if [ -n "${optimization_metrics}" ]; then
        log_success "Optimization metrics found: ${optimization_metrics}"
    else
        log_warning "No optimization metrics found"
    fi
    
    # Get speed improvement
    local speed_improvement
    speed_improvement=$(kubectl logs -n "${NAMESPACE}" -l app=truthgpt-enterprise-optimizer --tail=100 | grep "speed_improvement" | tail -1)
    
    if [ -n "${speed_improvement}" ]; then
        log_success "Speed improvement: ${speed_improvement}"
    else
        log_warning "No speed improvement metrics found"
    fi
    
    # Get memory reduction
    local memory_reduction
    memory_reduction=$(kubectl logs -n "${NAMESPACE}" -l app=truthgpt-enterprise-optimizer --tail=100 | grep "memory_reduction" | tail -1)
    
    if [ -n "${memory_reduction}" ]; then
        log_success "Memory reduction: ${memory_reduction}"
    else
        log_warning "No memory reduction metrics found"
    fi
    
    # Get energy efficiency
    local energy_efficiency
    energy_efficiency=$(kubectl logs -n "${NAMESPACE}" -l app=truthgpt-enterprise-optimizer --tail=100 | grep "energy_efficiency" | tail -1)
    
    if [ -n "${energy_efficiency}" ]; then
        log_success "Energy efficiency: ${energy_efficiency}"
    else
        log_warning "No energy efficiency metrics found"
    fi
}

# Check security metrics
check_security_metrics() {
    log_security "Checking enterprise security metrics..."
    
    # Get security metrics
    local security_metrics
    security_metrics=$(kubectl logs -n "${NAMESPACE}" -l app=truthgpt-enterprise-optimizer --tail=100 | grep "security" | tail -1)
    
    if [ -n "${security_metrics}" ]; then
        log_security "Security metrics found: ${security_metrics}"
    else
        log_warning "No security metrics found"
    fi
    
    # Get encryption strength
    local encryption_strength
    encryption_strength=$(kubectl logs -n "${NAMESPACE}" -l app=truthgpt-enterprise-optimizer --tail=100 | grep "encryption_strength" | tail -1)
    
    if [ -n "${encryption_strength}" ]; then
        log_security "Encryption strength: ${encryption_strength}"
    else
        log_warning "No encryption strength metrics found"
    fi
    
    # Get authentication score
    local authentication_score
    authentication_score=$(kubectl logs -n "${NAMESPACE}" -l app=truthgpt-enterprise-optimizer --tail=100 | grep "authentication_score" | tail -1)
    
    if [ -n "${authentication_score}" ]; then
        log_security "Authentication score: ${authentication_score}"
    else
        log_warning "No authentication score metrics found"
    fi
    
    # Get vulnerability score
    local vulnerability_score
    vulnerability_score=$(kubectl logs -n "${NAMESPACE}" -l app=truthgpt-enterprise-optimizer --tail=100 | grep "vulnerability_score" | tail -1)
    
    if [ -n "${vulnerability_score}" ]; then
        log_security "Vulnerability score: ${vulnerability_score}"
    else
        log_warning "No vulnerability score metrics found"
    fi
}

# Check compliance metrics
check_compliance_metrics() {
    log_enterprise "Checking enterprise compliance metrics..."
    
    # Get compliance metrics
    local compliance_metrics
    compliance_metrics=$(kubectl logs -n "${NAMESPACE}" -l app=truthgpt-enterprise-optimizer --tail=100 | grep "compliance" | tail -1)
    
    if [ -n "${compliance_metrics}" ]; then
        log_enterprise "Compliance metrics found: ${compliance_metrics}"
    else
        log_warning "No compliance metrics found"
    fi
    
    # Get GDPR compliance
    local gdpr_compliance
    gdpr_compliance=$(kubectl logs -n "${NAMESPACE}" -l app=truthgpt-enterprise-optimizer --tail=100 | grep "gdpr_compliance" | tail -1)
    
    if [ -n "${gdpr_compliance}" ]; then
        log_enterprise "GDPR compliance: ${gdpr_compliance}"
    else
        log_warning "No GDPR compliance metrics found"
    fi
    
    # Get SOX compliance
    local sox_compliance
    sox_compliance=$(kubectl logs -n "${NAMESPACE}" -l app=truthgpt-enterprise-optimizer --tail=100 | grep "sox_compliance" | tail -1)
    
    if [ -n "${sox_compliance}" ]; then
        log_enterprise "SOX compliance: ${sox_compliance}"
    else
        log_warning "No SOX compliance metrics found"
    fi
    
    # Get HIPAA compliance
    local hipaa_compliance
    hipaa_compliance=$(kubectl logs -n "${NAMESPACE}" -l app=truthgpt-enterprise-optimizer --tail=100 | grep "hipaa_compliance" | tail -1)
    
    if [ -n "${hipaa_compliance}" ]; then
        log_enterprise "HIPAA compliance: ${hipaa_compliance}"
    else
        log_warning "No HIPAA compliance metrics found"
    fi
}

# Check cost optimization
check_cost_optimization() {
    log_enterprise "Checking enterprise cost optimization..."
    
    # Get cost optimization metrics
    local cost_optimization
    cost_optimization=$(kubectl logs -n "${NAMESPACE}" -l app=truthgpt-enterprise-optimizer --tail=100 | grep "cost_optimization" | tail -1)
    
    if [ -n "${cost_optimization}" ]; then
        log_enterprise "Cost optimization metrics found: ${cost_optimization}"
    else
        log_warning "No cost optimization metrics found"
    fi
    
    # Get cost reduction
    local cost_reduction
    cost_reduction=$(kubectl logs -n "${NAMESPACE}" -l app=truthgpt-enterprise-optimizer --tail=100 | grep "cost_reduction" | tail -1)
    
    if [ -n "${cost_reduction}" ]; then
        log_enterprise "Cost reduction: ${cost_reduction}"
    else
        log_warning "No cost reduction metrics found"
    fi
    
    # Get resource efficiency
    local resource_efficiency
    resource_efficiency=$(kubectl logs -n "${NAMESPACE}" -l app=truthgpt-enterprise-optimizer --tail=100 | grep "resource_efficiency" | tail -1)
    
    if [ -n "${resource_efficiency}" ]; then
        log_enterprise "Resource efficiency: ${resource_efficiency}"
    else
        log_warning "No resource efficiency metrics found"
    fi
    
    # Get energy savings
    local energy_savings
    energy_savings=$(kubectl logs -n "${NAMESPACE}" -l app=truthgpt-enterprise-optimizer --tail=100 | grep "energy_savings" | tail -1)
    
    if [ -n "${energy_savings}" ]; then
        log_enterprise "Energy savings: ${energy_savings}"
    else
        log_warning "No energy savings metrics found"
    fi
}

# Check logs for errors
check_logs_for_errors() {
    log_info "Checking enterprise logs for errors..."
    
    local error_count=0
    local warning_count=0
    local critical_count=0
    
    # Check for ERROR level logs
    local error_logs
    error_logs=$(kubectl logs -n "${NAMESPACE}" -l app=truthgpt-enterprise-optimizer --tail=1000 | grep -i "error" | wc -l)
    
    if [ "${error_logs}" -gt 0 ]; then
        log_warning "Found ${error_logs} error logs"
        error_count=$((error_count + error_logs))
    fi
    
    # Check for WARNING level logs
    local warning_logs
    warning_logs=$(kubectl logs -n "${NAMESPACE}" -l app=truthgpt-enterprise-optimizer --tail=1000 | grep -i "warning" | wc -l)
    
    if [ "${warning_logs}" -gt 0 ]; then
        log_warning "Found ${warning_logs} warning logs"
        warning_count=$((warning_count + warning_logs))
    fi
    
    # Check for CRITICAL level logs
    local critical_logs
    critical_logs=$(kubectl logs -n "${NAMESPACE}" -l app=truthgpt-enterprise-optimizer --tail=1000 | grep -i "critical" | wc -l)
    
    if [ "${critical_logs}" -gt 0 ]; then
        log_critical "Found ${critical_logs} critical logs"
        critical_count=$((critical_count + critical_logs))
    fi
    
    # Check for SECURITY level logs
    local security_logs
    security_logs=$(kubectl logs -n "${NAMESPACE}" -l app=truthgpt-enterprise-optimizer --tail=1000 | grep -i "security" | wc -l)
    
    if [ "${security_logs}" -gt 0 ]; then
        log_security "Found ${security_logs} security logs"
    fi
    
    # Check for ENTERPRISE level logs
    local enterprise_logs
    enterprise_logs=$(kubectl logs -n "${NAMESPACE}" -l app=truthgpt-enterprise-optimizer --tail=1000 | grep -i "enterprise" | wc -l)
    
    if [ "${enterprise_logs}" -gt 0 ]; then
        log_enterprise "Found ${enterprise_logs} enterprise logs"
    fi
    
    if [ "${error_count}" -eq 0 ] && [ "${warning_count}" -eq 0 ] && [ "${critical_count}" -eq 0 ]; then
        log_success "No errors found in logs"
    else
        log_warning "Total issues found: $((error_count + warning_count + critical_count))"
    fi
    
    return $((error_count + warning_count + critical_count))
}

# Send enterprise alert
send_enterprise_alert() {
    local alert_type="$1"
    local message="$2"
    local severity="$3"
    
    log_info "Sending ${alert_type} alert: ${message}"
    
    # Send email alert
    if [ -n "${ALERT_EMAIL}" ]; then
        echo "${message}" | mail -s "TruthGPT ${alert_type} Alert - ${severity}" "${ALERT_EMAIL}"
    fi
    
    # Send Slack alert
    if [ -n "${SLACK_WEBHOOK}" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🚨 TruthGPT ${alert_type} Alert - ${severity}: ${message}\"}" \
            "${SLACK_WEBHOOK}"
    fi
    
    # Send Teams alert
    if [ -n "${TEAMS_WEBHOOK}" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🚨 TruthGPT ${alert_type} Alert - ${severity}: ${message}\"}" \
            "${TEAMS_WEBHOOK}"
    fi
}

# Generate enterprise monitoring report
generate_enterprise_monitoring_report() {
    log_info "Generating enterprise monitoring report..."
    
    local report_file="/tmp/truthgpt-enterprise-monitoring-report-$(date +%Y%m%d-%H%M%S).txt"
    
    {
        echo "=== Enterprise TruthGPT Monitoring Report ==="
        echo "Generated: $(date)"
        echo "Cluster: ${CLUSTER_NAME}"
        echo "Namespace: ${NAMESPACE}"
        echo "Region: ${AZURE_REGION}"
        echo "Resource Group: ${RESOURCE_GROUP}"
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
        
        echo "=== VPA Status ==="
        kubectl get vpa -n "${NAMESPACE}" -o wide
        echo ""
        
        echo "=== PDB Status ==="
        kubectl get pdb -n "${NAMESPACE}" -o wide
        echo ""
        
        echo "=== Resource Usage ==="
        kubectl top nodes
        kubectl top pods -n "${NAMESPACE}"
        echo ""
        
        echo "=== Recent Logs ==="
        kubectl logs -n "${NAMESPACE}" -l app=truthgpt-enterprise-optimizer --tail=50
        echo ""
        
        echo "=== Security Metrics ==="
        kubectl logs -n "${NAMESPACE}" -l app=truthgpt-enterprise-optimizer --tail=100 | grep -i "security" | tail -10
        echo ""
        
        echo "=== Compliance Metrics ==="
        kubectl logs -n "${NAMESPACE}" -l app=truthgpt-enterprise-optimizer --tail=100 | grep -i "compliance" | tail -10
        echo ""
        
        echo "=== Cost Optimization Metrics ==="
        kubectl logs -n "${NAMESPACE}" -l app=truthgpt-enterprise-optimizer --tail=100 | grep -i "cost" | tail -10
        echo ""
        
    } > "${report_file}"
    
    log_success "Enterprise monitoring report generated: ${report_file}"
    echo "${report_file}"
}

# Main enterprise monitoring function
main_enterprise_monitor() {
    log_info "Starting Enterprise TruthGPT monitoring..."
    
    local alerts_sent=0
    local critical_alerts=0
    
    # Check prerequisites
    check_prerequisites
    
    # Get cluster status
    get_cluster_status
    
    # Check pod health
    if ! check_pod_health; then
        send_enterprise_alert "CRITICAL" "Pod health check failed" "CRITICAL"
        alerts_sent=$((alerts_sent + 1))
        critical_alerts=$((critical_alerts + 1))
    fi
    
    # Check resource usage
    check_resource_usage
    
    # Check application health
    if ! check_application_health; then
        send_enterprise_alert "CRITICAL" "Application health check failed" "CRITICAL"
        alerts_sent=$((alerts_sent + 1))
        critical_alerts=$((critical_alerts + 1))
    fi
    
    # Check optimization performance
    check_optimization_performance
    
    # Check security metrics
    check_security_metrics
    
    # Check compliance metrics
    check_compliance_metrics
    
    # Check cost optimization
    check_cost_optimization
    
    # Check logs for errors
    local error_count
    if ! check_logs_for_errors; then
        error_count=$?
        if [ "${error_count}" -gt 10 ]; then
            send_enterprise_alert "WARNING" "High error count in logs: ${error_count}" "WARNING"
            alerts_sent=$((alerts_sent + 1))
        fi
    fi
    
    # Generate enterprise monitoring report
    local report_file
    report_file=$(generate_enterprise_monitoring_report)
    
    # Summary
    log_info "Enterprise monitoring completed"
    log_info "Alerts sent: ${alerts_sent}"
    log_info "Critical alerts: ${critical_alerts}"
    log_info "Report file: ${report_file}"
    
    if [ "${alerts_sent}" -eq 0 ]; then
        log_success "All enterprise systems are healthy"
    else
        log_warning "Some issues detected, alerts sent"
    fi
    
    if [ "${critical_alerts}" -gt 0 ]; then
        log_critical "Critical issues detected, immediate attention required"
    fi
}

# Continuous enterprise monitoring
continuous_enterprise_monitor() {
    log_info "Starting continuous enterprise monitoring..."
    
    while true; do
        main_enterprise_monitor
        sleep 300  # Check every 5 minutes
    done
}

# Parse command line arguments
main() {
    case "${1:-monitor}" in
        "monitor")
            main_enterprise_monitor
            ;;
        "continuous")
            continuous_enterprise_monitor
            ;;
        "report")
            generate_enterprise_monitoring_report
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
        "security")
            check_security_metrics
            ;;
        "compliance")
            check_compliance_metrics
            ;;
        "cost")
            check_cost_optimization
            ;;
        "help")
            echo "Usage: $0 [monitor|continuous|report|health|resources|logs|security|compliance|cost|help]"
            echo "  monitor     - Run single enterprise monitoring check"
            echo "  continuous  - Run continuous enterprise monitoring"
            echo "  report      - Generate enterprise monitoring report"
            echo "  health      - Check application health"
            echo "  resources   - Check resource usage"
            echo "  logs        - Check logs for errors"
            echo "  security    - Check security metrics"
            echo "  compliance  - Check compliance metrics"
            echo "  cost        - Check cost optimization"
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









