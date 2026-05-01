"""
⚡ Performance Tuner
Automated performance tuning and optimization recommendations
"""

import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    requests_per_second: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    cache_hit_rate: float
    queue_depth: int
    active_batches: int
    cpu_usage: Optional[float] = None
    memory_usage_mb: Optional[float] = None


@dataclass
class TuningRecommendation:
    """Performance tuning recommendation"""
    category: str
    issue: str
    recommendation: str
    priority: str  # high, medium, low
    expected_impact: str


class PerformanceTuner:
    """Performance tuning and optimization tool"""
    
    def __init__(self, api_url: str = "http://localhost:8080"):
        self.api_url = api_url.rstrip("/")
        self.client = httpx.Client(timeout=10.0)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Fetch current metrics from API"""
        try:
            response = self.client.get(f"{self.api_url}/metrics")
            response.raise_for_status()
            
            # Parse Prometheus format
            metrics = {}
            for line in response.text.split("\n"):
                if line and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            metrics[parts[0]] = float(parts[1])
                        except ValueError:
                            pass
            
            return metrics
        
        except Exception as e:
            console.print(f"[red]Failed to fetch metrics: {e}[/red]")
            return {}
    
    def analyze_health(self) -> Dict[str, Any]:
        """Analyze API health"""
        try:
            response = self.client.get(f"{self.api_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception:
            return {}
    
    def calculate_performance_metrics(self, metrics: Dict[str, Any]) -> PerformanceMetrics:
        """Calculate performance metrics from raw Prometheus data"""
        # Extract key metrics (simplified - would need proper parsing)
        return PerformanceMetrics(
            requests_per_second=metrics.get("rate(inference_requests_total[5m])", 0.0),
            avg_latency_ms=metrics.get("inference_request_duration_ms", 0.0),
            p95_latency_ms=metrics.get("inference_request_duration_ms_p95", 0.0),
            p99_latency_ms=metrics.get("inference_request_duration_ms_p99", 0.0),
            error_rate=metrics.get("inference_errors_5xx_total", 0.0) / max(metrics.get("inference_requests_total", 1), 1),
            cache_hit_rate=metrics.get("inference_cache_hits_total", 0.0) / max(
                metrics.get("inference_cache_hits_total", 0.0) + metrics.get("inference_cache_misses_total", 0.0), 1
            ),
            queue_depth=int(metrics.get("inference_queue_depth", 0)),
            active_batches=int(metrics.get("inference_active_batches", 0)),
            cpu_usage=metrics.get("process_cpu_percent"),
            memory_usage_mb=metrics.get("process_memory_bytes", 0) / 1024 / 1024
        )
    
    def generate_recommendations(
        self,
        metrics: PerformanceMetrics,
        config: Dict[str, Any]
    ) -> List[TuningRecommendation]:
        """Generate performance tuning recommendations"""
        recommendations = []
        
        # Latency recommendations
        if metrics.p95_latency_ms > 600:
            recommendations.append(TuningRecommendation(
                category="Latency",
                issue=f"High p95 latency ({metrics.p95_latency_ms:.2f}ms)",
                recommendation="Consider: 1) Increase batch size, 2) Enable caching, 3) Scale horizontally",
                priority="high",
                expected_impact="Reduce latency by 20-40%"
            ))
        
        # Cache recommendations
        if metrics.cache_hit_rate < 0.3:
            recommendations.append(TuningRecommendation(
                category="Caching",
                issue=f"Low cache hit rate ({metrics.cache_hit_rate*100:.1f}%)",
                recommendation="Consider: 1) Increase cache TTL, 2) Enable Redis, 3) Normalize prompts for better cache key matching",
                priority="medium",
                expected_impact="Reduce latency by 50-80% for cached requests"
            ))
        
        # Queue recommendations
        if metrics.queue_depth > 50:
            recommendations.append(TuningRecommendation(
                category="Capacity",
                issue=f"High queue depth ({metrics.queue_depth})",
                recommendation="Consider: 1) Increase batch timeout, 2) Scale workers, 3) Increase batch size",
                priority="high",
                expected_impact="Reduce queue wait time by 30-50%"
            ))
        
        # Batch size recommendations
        batch_size = config.get("BATCH_MAX_SIZE", 32)
        if metrics.active_batches < 2 and metrics.requests_per_second > 50:
            recommendations.append(TuningRecommendation(
                category="Batching",
                issue="Batches may be too large or timeout too short",
                recommendation=f"Consider: 1) Reduce batch size to {batch_size // 2}, 2) Reduce flush timeout to 10ms",
                priority="medium",
                expected_impact="Improve latency for low-traffic periods"
            ))
        
        # Error rate recommendations
        if metrics.error_rate > 0.01:
            recommendations.append(TuningRecommendation(
                category="Reliability",
                issue=f"High error rate ({metrics.error_rate*100:.2f}%)",
                recommendation="Consider: 1) Check circuit breakers, 2) Review error logs, 3) Scale resources",
                priority="high",
                expected_impact="Improve reliability and user experience"
            ))
        
        # Memory recommendations
        if metrics.memory_usage_mb and metrics.memory_usage_mb > 8000:
            recommendations.append(TuningRecommendation(
                category="Resources",
                issue=f"High memory usage ({metrics.memory_usage_mb:.0f}MB)",
                recommendation="Consider: 1) Reduce batch size, 2) Enable model quantization, 3) Scale vertically",
                priority="medium",
                expected_impact="Prevent OOM errors and improve stability"
            ))
        
        # CPU recommendations
        if metrics.cpu_usage and metrics.cpu_usage > 80:
            recommendations.append(TuningRecommendation(
                category="Resources",
                issue=f"High CPU usage ({metrics.cpu_usage:.1f}%)",
                recommendation="Consider: 1) Scale horizontally, 2) Optimize code paths, 3) Use async I/O",
                priority="medium",
                expected_impact="Improve throughput and reduce latency"
            ))
        
        return recommendations
    
    def print_analysis(
        self,
        metrics: PerformanceMetrics,
        recommendations: List[TuningRecommendation]
    ):
        """Print performance analysis"""
        # Metrics panel
        metrics_panel = Panel(
            f"[bold]Throughput:[/bold] {metrics.requests_per_second:.2f} req/s\n"
            f"[bold]Avg Latency:[/bold] {metrics.avg_latency_ms:.2f}ms\n"
            f"[bold]p95 Latency:[/bold] {metrics.p95_latency_ms:.2f}ms\n"
            f"[bold]p99 Latency:[/bold] {metrics.p99_latency_ms:.2f}ms\n"
            f"[bold]Error Rate:[/bold] {metrics.error_rate*100:.2f}%\n"
            f"[bold]Cache Hit Rate:[/bold] {metrics.cache_hit_rate*100:.1f}%\n"
            f"[bold]Queue Depth:[/bold] {metrics.queue_depth}\n"
            f"[bold]Active Batches:[/bold] {metrics.active_batches}",
            title="📊 Current Performance",
            border_style="blue"
        )
        console.print(metrics_panel)
        
        # Recommendations table
        if recommendations:
            table = Table(title="🎯 Tuning Recommendations")
            table.add_column("Priority", style="red" if any(r.priority == "high" for r in recommendations) else "yellow")
            table.add_column("Category", style="cyan")
            table.add_column("Issue", style="white")
            table.add_column("Recommendation", style="green")
            table.add_column("Expected Impact", style="yellow")
            
            # Sort by priority
            priority_order = {"high": 0, "medium": 1, "low": 2}
            sorted_recs = sorted(recommendations, key=lambda r: priority_order.get(r.priority, 3))
            
            for rec in sorted_recs:
                table.add_row(
                    rec.priority.upper(),
                    rec.category,
                    rec.issue,
                    rec.recommendation,
                    rec.expected_impact
                )
            
            console.print(table)
        else:
            console.print("[green]✓ No major issues detected. Performance looks good![/green]")
    
    def generate_config_suggestions(
        self,
        metrics: PerformanceMetrics,
        current_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate suggested configuration changes"""
        suggestions = {}
        
        # Batch size suggestions
        if metrics.queue_depth > 50:
            suggestions["BATCH_MAX_SIZE"] = min(current_config.get("BATCH_MAX_SIZE", 32) * 2, 64)
            suggestions["BATCH_FLUSH_TIMEOUT_MS"] = max(current_config.get("BATCH_FLUSH_TIMEOUT_MS", 20) - 5, 10)
        
        # Rate limit suggestions
        if metrics.error_rate > 0.05:
            current_rpm = current_config.get("RATE_LIMIT_RPM", 600)
            suggestions["RATE_LIMIT_RPM"] = max(current_rpm * 0.8, 300)
        
        return suggestions
    
    def tune(self):
        """Run performance tuning analysis"""
        console.print("[bold]Running performance analysis...[/bold]\n")
        
        # Get current metrics
        raw_metrics = self.get_metrics()
        if not raw_metrics:
            console.print("[red]Could not fetch metrics. Is the API running?[/red]")
            return
        
        # Calculate performance metrics
        perf_metrics = self.calculate_performance_metrics(raw_metrics)
        
        # Current config (would be fetched from environment or config)
        current_config = {
            "BATCH_MAX_SIZE": 32,
            "BATCH_FLUSH_TIMEOUT_MS": 20,
            "RATE_LIMIT_RPM": 600,
        }
        
        # Generate recommendations
        recommendations = self.generate_recommendations(perf_metrics, current_config)
        
        # Print analysis
        self.print_analysis(perf_metrics, recommendations)
        
        # Generate config suggestions
        suggestions = self.generate_config_suggestions(perf_metrics, current_config)
        if suggestions:
            console.print("\n[bold]Suggested Configuration Changes:[/bold]")
            for key, value in suggestions.items():
                console.print(f"  {key}={value}")


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Tuning Tool")
    parser.add_argument("--url", default="http://localhost:8080", help="API URL")
    
    args = parser.parse_args()
    
    tuner = PerformanceTuner(args.url)
    tuner.tune()


if __name__ == "__main__":
    main()

