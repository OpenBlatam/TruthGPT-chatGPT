#!/bin/bash
# 📊 Monitoring Script
# Real-time monitoring and alerting for inference API

set -e

# Configuration
API_URL=${API_URL:-http://localhost:8080}
CHECK_INTERVAL=${CHECK_INTERVAL:-10}
ALERT_EMAIL=${ALERT_EMAIL:-}
SLACK_WEBHOOK=${SLACK_WEBHOOK:-}

# Thresholds
LATENCY_THRESHOLD=600
ERROR_RATE_THRESHOLD=0.02
QUEUE_DEPTH_THRESHOLD=100

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

check_health() {
    local status=$(curl -s -o /dev/null -w "%{http_code}" ${API_URL}/health 2>/dev/null || echo "000")
    echo $status
}

get_metric() {
    local metric_name=$1
    curl -s ${API_URL}/metrics | grep "^${metric_name}" | awk '{print $2}' | head -1
}

check_latency() {
    local p95=$(get_metric "inference_request_duration_ms_p95")
    if [ ! -z "$p95" ] && (( $(echo "$p95 > $LATENCY_THRESHOLD" | bc -l) )); then
        alert "HIGH_LATENCY" "p95 latency is ${p95}ms (threshold: ${LATENCY_THRESHOLD}ms)"
        return 1
    fi
    return 0
}

check_errors() {
    local total=$(get_metric "inference_requests_total")
    local errors=$(get_metric "inference_errors_5xx_total")
    
    if [ ! -z "$total" ] && [ ! -z "$errors" ] && [ "$total" != "0" ]; then
        local rate=$(echo "scale=4; $errors / $total" | bc)
        if (( $(echo "$rate > $ERROR_RATE_THRESHOLD" | bc -l) )); then
            alert "HIGH_ERROR_RATE" "Error rate is $(echo "$rate * 100" | bc)% (threshold: $(echo "$ERROR_RATE_THRESHOLD * 100" | bc)%)"
            return 1
        fi
    fi
    return 0
}

check_queue() {
    local depth=$(get_metric "inference_queue_depth")
    if [ ! -z "$depth" ] && [ "$depth" -gt "$QUEUE_DEPTH_THRESHOLD" ]; then
        alert "HIGH_QUEUE_DEPTH" "Queue depth is ${depth} (threshold: ${QUEUE_DEPTH_THRESHOLD})"
        return 1
    fi
    return 0
}

alert() {
    local severity=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo -e "${RED}[ALERT]${NC} ${timestamp} - ${severity}: ${message}"
    
    # Email alert
    if [ ! -z "$ALERT_EMAIL" ]; then
        echo "${message}" | mail -s "Inference API Alert: ${severity}" "$ALERT_EMAIL" 2>/dev/null || true
    fi
    
    # Slack alert
    if [ ! -z "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🚨 Inference API Alert: ${severity}\n${message}\"}" \
            "$SLACK_WEBHOOK" 2>/dev/null || true
    fi
}

display_stats() {
    clear
    echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║   Inference API Monitor                ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
    echo ""
    
    # Health
    local health_status=$(check_health)
    if [ "$health_status" == "200" ]; then
        echo -e "${GREEN}✓${NC} Health: OK"
    else
        echo -e "${RED}✗${NC} Health: FAILED (HTTP ${health_status})"
    fi
    
    # Metrics
    local rps=$(get_metric "rate(inference_requests_total[5m])")
    local p95=$(get_metric "inference_request_duration_ms_p95")
    local errors=$(get_metric "inference_errors_5xx_total")
    local queue=$(get_metric "inference_queue_depth")
    local cache_hits=$(get_metric "inference_cache_hits_total")
    local cache_misses=$(get_metric "inference_cache_misses_total")
    
    echo ""
    echo -e "${YELLOW}Metrics:${NC}"
    echo "  Requests/sec: ${rps:-N/A}"
    echo "  p95 Latency:  ${p95:-N/A}ms"
    echo "  5xx Errors:   ${errors:-0}"
    echo "  Queue Depth:  ${queue:-0}"
    
    if [ ! -z "$cache_hits" ] && [ ! -z "$cache_misses" ]; then
        local total_cache=$((cache_hits + cache_misses))
        if [ "$total_cache" -gt 0 ]; then
            local hit_rate=$(echo "scale=2; $cache_hits * 100 / $total_cache" | bc)
            echo "  Cache Hit Rate: ${hit_rate}%"
        fi
    fi
    
    echo ""
    echo -e "${YELLOW}Last Update:${NC} $(date '+%Y-%m-%d %H:%M:%S')"
}

main() {
    echo "Starting monitoring (interval: ${CHECK_INTERVAL}s)"
    echo "Press Ctrl+C to stop"
    echo ""
    
    while true; do
        display_stats
        
        # Run checks
        check_latency
        check_errors
        check_queue
        
        sleep $CHECK_INTERVAL
    done
}

# Trap Ctrl+C
trap 'echo -e "\n${YELLOW}Monitoring stopped${NC}"; exit 0' INT

main


