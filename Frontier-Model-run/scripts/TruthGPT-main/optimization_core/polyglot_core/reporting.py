"""
Reporting utilities for polyglot_core.

Generates comprehensive reports from benchmarks, profiling, and metrics.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import json
import csv


@dataclass
class ReportSection:
    """Section of a report."""
    title: str
    content: str
    data: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceReport:
    """Complete performance report."""
    title: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    sections: List[ReportSection] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def add_section(self, title: str, content: str, data: Optional[Dict] = None):
        """Add a section to the report."""
        self.sections.append(ReportSection(title, content, data))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'title': self.title,
            'timestamp': self.timestamp,
            'summary': self.summary,
            'sections': [
                {
                    'title': s.title,
                    'content': s.content,
                    'data': s.data
                }
                for s in self.sections
            ]
        }
    
    def to_json(self, filepath: Optional[str] = None) -> str:
        """Export as JSON."""
        json_str = json.dumps(self.to_dict(), indent=2)
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
        return json_str
    
    def to_markdown(self) -> str:
        """Export as Markdown."""
        lines = [
            f"# {self.title}",
            "",
            f"**Generated:** {self.timestamp}",
            "",
        ]
        
        if self.summary:
            lines.append("## Summary")
            lines.append("")
            for key, value in self.summary.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")
        
        for section in self.sections:
            lines.append(f"## {section.title}")
            lines.append("")
            lines.append(section.content)
            lines.append("")
            
            if section.data:
                lines.append("### Data")
                lines.append("")
                lines.append("```json")
                lines.append(json.dumps(section.data, indent=2))
                lines.append("```")
                lines.append("")
        
        return "\n".join(lines)
    
    def to_html(self) -> str:
        """Export as HTML."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{self.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
        pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>{self.title}</h1>
    <p><strong>Generated:</strong> {self.timestamp}</p>
"""
        
        if self.summary:
            html += '<div class="summary"><h2>Summary</h2><ul>'
            for key, value in self.summary.items():
                html += f'<li><strong>{key}:</strong> {value}</li>'
            html += '</ul></div>'
        
        for section in self.sections:
            html += f'<h2>{section.title}</h2>'
            html += f'<p>{section.content.replace(chr(10), "<br>")}</p>'
            
            if section.data:
                html += '<h3>Data</h3>'
                html += '<pre>'
                html += json.dumps(section.data, indent=2)
                html += '</pre>'
        
        html += """
</body>
</html>
"""
        return html
    
    def save(self, filepath: str, format: str = "markdown"):
        """
        Save report to file.
        
        Args:
            filepath: Output file path
            format: Format ("markdown", "html", "json")
        """
        path = Path(filepath)
        
        if format == "markdown":
            content = self.to_markdown()
            path = path.with_suffix('.md')
        elif format == "html":
            content = self.to_html()
            path = path.with_suffix('.html')
        elif format == "json":
            content = self.to_json()
            path = path.with_suffix('.json')
        else:
            raise ValueError(f"Unknown format: {format}")
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Report saved to: {path}")


class ReportGenerator:
    """
    Generates comprehensive reports from benchmarks and metrics.
    
    Example:
        >>> generator = ReportGenerator()
        >>> report = generator.generate_benchmark_report(benchmark_results)
        >>> report.save("benchmark_report", format="html")
    """
    
    def generate_benchmark_report(
        self,
        results: Dict[str, Any],
        title: str = "Benchmark Report"
    ) -> PerformanceReport:
        """
        Generate report from benchmark results.
        
        Args:
            results: Benchmark results dictionary
            title: Report title
            
        Returns:
            PerformanceReport
        """
        report = PerformanceReport(title=title)
        
        # Summary
        if results:
            successful = [r for r in results.values() if getattr(r, 'success', True)]
            if successful:
                fastest = max(successful, key=lambda x: getattr(x, 'throughput', 0))
                report.summary = {
                    'Total benchmarks': len(results),
                    'Successful': len(successful),
                    'Fastest backend': getattr(fastest, 'backend', 'unknown'),
                    'Max throughput': f"{getattr(fastest, 'throughput', 0):.0f} ops/s"
                }
        
        # Backend comparison
        backend_data = {}
        for name, result in results.items():
            backend = getattr(result, 'backend', 'unknown')
            if backend not in backend_data:
                backend_data[backend] = []
            backend_data[backend].append({
                'name': name,
                'throughput': getattr(result, 'throughput', 0),
                'avg_time_ms': getattr(result, 'avg_time_ms', 0)
            })
        
        report.add_section(
            "Backend Comparison",
            "Performance comparison across different backends.",
            backend_data
        )
        
        # Detailed results
        detailed = {
            name: asdict(result) if hasattr(result, '__dict__') else result
            for name, result in results.items()
        }
        
        report.add_section(
            "Detailed Results",
            "Complete benchmark results for all operations.",
            detailed
        )
        
        return report
    
    def generate_profiling_report(
        self,
        profiler_summary: Dict[str, Any],
        title: str = "Profiling Report"
    ) -> PerformanceReport:
        """
        Generate report from profiling data.
        
        Args:
            profiler_summary: Profiler summary dictionary
            title: Report title
            
        Returns:
            PerformanceReport
        """
        report = PerformanceReport(title=title)
        
        if profiler_summary:
            report.summary = {
                'Total operations': len(profiler_summary),
                'Total runs': sum(s.get('runs', 0) for s in profiler_summary.values())
            }
        
        report.add_section(
            "Operation Performance",
            "Performance metrics for each operation.",
            profiler_summary
        )
        
        return report
    
    def generate_metrics_report(
        self,
        metrics_summaries: Dict[str, Any],
        title: str = "Metrics Report"
    ) -> PerformanceReport:
        """
        Generate report from metrics data.
        
        Args:
            metrics_summaries: Metrics summaries dictionary
            title: Report title
            
        Returns:
            PerformanceReport
        """
        report = PerformanceReport(title=title)
        
        if metrics_summaries:
            total_metrics = len(metrics_summaries)
            total_samples = sum(
                s.get('count', 0) for s in metrics_summaries.values()
                if isinstance(s, dict)
            )
            
            report.summary = {
                'Total metrics': total_metrics,
                'Total samples': total_samples
            }
        
        report.add_section(
            "Metrics Summary",
            "Aggregated metrics statistics.",
            metrics_summaries
        )
        
        return report


def generate_benchmark_report(
    results: Dict[str, Any],
    output_file: str,
    format: str = "html"
) -> PerformanceReport:
    """Convenience function to generate and save benchmark report."""
    generator = ReportGenerator()
    report = generator.generate_benchmark_report(results)
    report.save(output_file, format)
    return report













