"""
📊 Prometheus Metrics Collector
Enterprise-grade metrics collection for inference API with percentiles and histograms
"""

import time
from typing import Dict, Optional, List
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import Lock
import threading


@dataclass
class MetricsSnapshot:
    """Metrics snapshot with percentiles"""
    requests_total: int = 0
    requests_5xx: int = 0
    requests_4xx: int = 0
    request_duration_ms: float = 0.0
    request_duration_p50_ms: float = 0.0
    request_duration_p95_ms: float = 0.0
    request_duration_p99_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    queue_depth: int = 0
    active_batches: int = 0
    circuit_breaker_open: int = 0
    rate_limit_hits: int = 0
    throughput_rps: float = 0.0
    active_connections: int = 0
    batch_size_avg: float = 0.0
    tokens_per_second: float = 0.0


class MetricsCollector:
    """Prometheus-style metrics collector with histogram support"""
    
    def __init__(self, max_history_size: int = 10000):
        self._lock = Lock()
        self.max_history_size = max_history_size
        self._reset()
        self._histogram_buckets = [
            1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000
        ]
        self._request_times: deque = deque(maxlen=1000)  # For throughput calculation
    
    def _reset(self):
        """Reset all metrics"""
        with self._lock:
            self.counters: Dict[str, int] = defaultdict(int)
            self.gauges: Dict[str, float] = defaultdict(float)
            self.histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_history_size))
            self.labels_cache: Dict[str, Dict[str, str]] = {}
            self.start_time = time.time()
            self.last_reset_time = time.time()
    
    def increment(self, name: str, labels: Optional[Dict[str, str]] = None, value: int = 1):
        """Increment counter"""
        with self._lock:
            key = self._make_key(name, labels)
            self.counters[key] += value
            if labels:
                self.labels_cache[key] = labels
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set gauge value"""
        with self._lock:
            key = self._make_key(name, labels)
            self.gauges[key] = value
            if labels:
                self.labels_cache[key] = labels
    
    def observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe histogram value"""
        with self._lock:
            key = self._make_key(name, labels)
            self.histograms[key].append(value)
            if labels:
                self.labels_cache[key] = labels
    
    def record_request_time(self, duration_ms: float):
        """Record request time for throughput calculation"""
        with self._lock:
            self._request_times.append(time.time())
            # Keep only last minute
            cutoff = time.time() - 60
            while self._request_times and self._request_times[0] < cutoff:
                self._request_times.popleft()
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Make metric key with labels"""
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f'{name}{{{label_str}}}'
    
    def _calculate_percentile(self, values: deque, percentile: float) -> float:
        """Calculate percentile from deque"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        if not sorted_values:
            return 0.0
        index = int(len(sorted_values) * percentile / 100.0)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def _calculate_avg(self, values: deque) -> float:
        """Calculate average"""
        if not values:
            return 0.0
        return sum(values) / len(values)
    
    def get_snapshot(self) -> MetricsSnapshot:
        """Get comprehensive metrics snapshot"""
        with self._lock:
            duration_values = self.histograms.get("inference_request_duration_ms", deque())
            batch_sizes = self.histograms.get("inference_batch_size", deque())
            
            # Calculate throughput (requests per second)
            throughput_rps = len(self._request_times) / 60.0 if self._request_times else 0.0
            
            return MetricsSnapshot(
                requests_total=self.counters.get("inference_requests_total", 0),
                requests_5xx=self.counters.get("inference_errors_5xx_total", 0),
                requests_4xx=self.counters.get("inference_errors_4xx_total", 0),
                request_duration_ms=self._calculate_avg(duration_values),
                request_duration_p50_ms=self._calculate_percentile(duration_values, 50),
                request_duration_p95_ms=self._calculate_percentile(duration_values, 95),
                request_duration_p99_ms=self._calculate_percentile(duration_values, 99),
                cache_hits=self.counters.get("inference_cache_hits_total", 0),
                cache_misses=self.counters.get("inference_cache_misses_total", 0),
                queue_depth=int(self.gauges.get("inference_queue_depth", 0)),
                active_batches=int(self.gauges.get("inference_active_batches", 0)),
                circuit_breaker_open=self.counters.get("circuit_breaker_open_total", 0),
                rate_limit_hits=self.counters.get("rate_limit_hits_total", 0),
                throughput_rps=throughput_rps,
                active_connections=int(self.gauges.get("inference_active_connections", 0)),
                batch_size_avg=self._calculate_avg(batch_sizes),
                tokens_per_second=self.gauges.get("inference_tokens_per_second", 0.0),
            )
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        with self._lock:
            lines = []
            
            # Counters
            for key, value in sorted(self.counters.items()):
                name = key.split("{")[0]
                if name not in [l.split("{")[0] for l in lines if "# TYPE" in l]:
                    lines.append(f"# HELP {name} Total count")
                    lines.append(f"# TYPE {name} counter")
                lines.append(f"{key} {value}")
            
            # Gauges
            for key, value in sorted(self.gauges.items()):
                name = key.split("{")[0]
                if name not in [l.split("{")[0] for l in lines if "# TYPE" in l]:
                    lines.append(f"# HELP {name} Current value")
                    lines.append(f"# TYPE {name} gauge")
                lines.append(f"{key} {value:.6f}")
            
            # Histograms
            for key, values in sorted(self.histograms.items()):
                if not values:
                    continue
                name = key.split("{")[0]
                base_name = name.replace("_bucket", "").replace("_sum", "").replace("_count", "")
                
                if f"# TYPE {base_name}" not in "\n".join(lines):
                    lines.append(f"# HELP {base_name} Histogram of values")
                    lines.append(f"# TYPE {base_name} histogram")
                
                # Calculate buckets
                sorted_values = sorted(values)
                for bucket in self._histogram_buckets:
                    count = sum(1 for v in sorted_values if v <= bucket)
                    # Extract labels from key
                    labels_part = key.split("{")[1] if "{" in key else ""
                    bucket_key = f'{base_name}_bucket{{{labels_part}le="{bucket:.0f}"}}' if labels_part else f'{base_name}_bucket{{le="{bucket:.0f}"}}'
                    lines.append(f"{bucket_key} {count}")
                
                # Sum and count
                total_sum = sum(sorted_values)
                count = len(sorted_values)
                labels_part = key.split("{")[1] if "{" in key else ""
                if labels_part:
                    sum_key = f'{base_name}_sum{{{labels_part}}}'
                    count_key = f'{base_name}_count{{{labels_part}}}'
                else:
                    sum_key = f'{base_name}_sum'
                    count_key = f'{base_name}_count'
                lines.append(f"{sum_key} {total_sum:.6f}")
                lines.append(f"{count_key} {count}")
            
            # Uptime
            uptime = time.time() - self.start_time
            lines.append(f"# HELP process_uptime_seconds Process uptime in seconds")
            lines.append(f"# TYPE process_uptime_seconds gauge")
            lines.append(f"process_uptime_seconds {uptime:.6f}")
            
            # System metrics (optional - requires psutil)
            try:
                import psutil
                process = psutil.Process()
                lines.append(f"# HELP process_cpu_percent CPU usage percentage")
                lines.append(f"# TYPE process_cpu_percent gauge")
                lines.append(f"process_cpu_percent {process.cpu_percent():.2f}")
                
                lines.append(f"# HELP process_memory_bytes Memory usage in bytes")
                lines.append(f"# TYPE process_memory_bytes gauge")
                lines.append(f"process_memory_bytes {process.memory_info().rss}")
            except ImportError:
                # psutil not available, skip system metrics
                pass
            
            return "\n".join(lines) + "\n"
    
    def reset(self):
        """Reset all metrics (for testing)"""
        with self._lock:
            self._reset()


# Global metrics collector instance
metrics_collector = MetricsCollector()


