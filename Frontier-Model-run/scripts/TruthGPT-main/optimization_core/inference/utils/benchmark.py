"""
📊 Benchmarking Utilities
Performance testing and benchmarking tools for inference API
"""

import asyncio
import statistics
import time
from typing import List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn

console = Console()


@dataclass
class BenchmarkResult:
    """Benchmark result data"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    latencies: List[float] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    error_count: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: float = 0
    
    def __post_init__(self):
        if self.end_time == 0:
            self.end_time = time.time()
    
    @property
    def duration(self) -> float:
        """Total duration in seconds"""
        return self.end_time - self.start_time
    
    @property
    def requests_per_second(self) -> float:
        """Throughput in requests per second"""
        return self.successful_requests / max(self.duration, 0.001)
    
    @property
    def success_rate(self) -> float:
        """Success rate percentage"""
        return (self.successful_requests / max(self.total_requests, 1)) * 100
    
    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate percentage"""
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / max(total, 1)) * 100
    
    @property
    def latency_stats(self) -> Dict[str, float]:
        """Latency statistics"""
        if not self.latencies:
            return {}
        
        return {
            "min": min(self.latencies),
            "max": max(self.latencies),
            "mean": statistics.mean(self.latencies),
            "median": statistics.median(self.latencies),
            "p50": statistics.median(self.latencies),
            "p95": self._percentile(95),
            "p99": self._percentile(99),
            "stdev": statistics.stdev(self.latencies) if len(self.latencies) > 1 else 0,
        }
    
    def _percentile(self, p: int) -> float:
        """Calculate percentile"""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        index = int(len(sorted_latencies) * p / 100)
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]


class InferenceBenchmark:
    """Benchmark runner for inference API"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_token: str = "changeme",
        timeout: float = 30.0
    ):
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def run_single_request(
        self,
        prompt: str,
        model: str = "gpt-4o",
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Run a single inference request"""
        params = params or {}
        
        start = time.time()
        try:
            response = await self.client.post(
                f"{self.base_url}/v1/infer",
                headers={"Authorization": f"Bearer {self.api_token}"},
                json={
                    "model": model,
                    "prompt": prompt,
                    "params": params
                }
            )
            elapsed = time.time() - start
            
            response.raise_for_status()
            data = response.json()
            
            return {
                "success": True,
                "latency_ms": elapsed * 1000,
                "status_code": response.status_code,
                "cached": data.get("cached", False),
                "output_length": len(data.get("output", "")),
            }
        
        except Exception as e:
            elapsed = time.time() - start
            return {
                "success": False,
                "latency_ms": elapsed * 1000,
                "error": str(e),
            }
    
    async def run_benchmark(
        self,
        num_requests: int,
        concurrency: int = 10,
        prompt: str = "Hello, world!",
        model: str = "gpt-4o",
        params: Dict[str, Any] = None
    ) -> BenchmarkResult:
        """Run benchmark with specified parameters"""
        params = params or {}
        result = BenchmarkResult(
            total_requests=num_requests,
            successful_requests=0,
            failed_requests=0
        )
        
        semaphore = asyncio.Semaphore(concurrency)
        
        async def make_request():
            async with semaphore:
                req_result = await self.run_single_request(prompt, model, params)
                
                if req_result["success"]:
                    result.successful_requests += 1
                    result.latencies.append(req_result["latency_ms"])
                    
                    if req_result.get("cached"):
                        result.cache_hits += 1
                    else:
                        result.cache_misses += 1
                else:
                    result.failed_requests += 1
                    result.error_count += 1
        
        with Progress(
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("[green]{task.completed}/{task.total} requests")
        ) as progress:
            task = progress.add_task("Running benchmark...", total=num_requests)
            
            # Create all tasks
            tasks = [make_request() for _ in range(num_requests)]
            
            # Execute with progress tracking
            completed = 0
            for coro in asyncio.as_completed(tasks):
                await coro
                completed += 1
                progress.update(task, completed=completed)
        
        result.end_time = time.time()
        return result
    
    def print_results(self, result: BenchmarkResult):
        """Print benchmark results in a formatted table"""
        latency_stats = result.latency_stats
        
        # Summary panel
        summary = Panel(
            f"[bold]Duration:[/bold] {result.duration:.2f}s\n"
            f"[bold]Requests/sec:[/bold] {result.requests_per_second:.2f}\n"
            f"[bold]Success Rate:[/bold] {result.success_rate:.2f}%\n"
            f"[bold]Cache Hit Rate:[/bold] {result.cache_hit_rate:.2f}%",
            title="📊 Benchmark Summary",
            border_style="green"
        )
        console.print(summary)
        
        # Latency table
        if latency_stats:
            table = Table(title="⏱️ Latency Statistics (ms)")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="yellow", justify="right")
            
            for metric, value in latency_stats.items():
                if metric == "stdev":
                    table.add_row(metric.upper(), f"{value:.2f}")
                else:
                    table.add_row(metric.upper(), f"{value:.2f}")
            
            console.print(table)
        
        # Error summary
        if result.error_count > 0:
            console.print(f"[red]⚠ Errors: {result.error_count}[/red]")
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


async def main():
    """Main benchmark function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Inference API")
    parser.add_argument("--url", default="http://localhost:8080", help="API URL")
    parser.add_argument("--token", default="changeme", help="API token")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrency level")
    parser.add_argument("--prompt", default="Hello, world!", help="Test prompt")
    parser.add_argument("--model", default="gpt-4o", help="Model name")
    
    args = parser.parse_args()
    
    benchmark = InferenceBenchmark(args.url, args.token)
    
    try:
        console.print(f"[bold]Starting benchmark:[/bold] {args.requests} requests, {args.concurrency} concurrent")
        
        result = await benchmark.run_benchmark(
            num_requests=args.requests,
            concurrency=args.concurrency,
            prompt=args.prompt,
            model=args.model
        )
        
        benchmark.print_results(result)
    
    finally:
        await benchmark.close()


if __name__ == "__main__":
    asyncio.run(main())



