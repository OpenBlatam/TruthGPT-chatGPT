"""
Test Reporting Framework
Comprehensive reporting and visualization for test execution
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import os

class TestReporter:
    """Comprehensive test reporting and visualization."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def print_comprehensive_report(self, report: Dict[str, Any]):
        """Print comprehensive test report."""
        print("\n" + "="*140)
        print("🚀 ENHANCED OPTIMIZATION CORE TEST REPORT V4")
        print("="*140)
        
        summary = report['summary']
        print(f"\n📊 OVERALL SUMMARY:")
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  Passed: {summary['passed']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Errors: {summary['errors']}")
        print(f"  Skipped: {summary['skipped']}")
        print(f"  Timeouts: {summary['timeouts']}")
        print(f"  Success Rate: {summary['success_rate']:.1f}%")
        print(f"  Total Time: {summary['total_execution_time']:.2f}s")
        print(f"  Total Memory: {summary['total_memory_usage']:.2f}MB")
        
        print(f"\n📈 CATEGORY BREAKDOWN:")
        for category, stats in report['category_stats'].items():
            category_success_rate = (stats['passed'] / stats['tests'] * 100) if stats['tests'] > 0 else 0
            print(f"  {category.upper()}:")
            print(f"    Tests: {stats['tests']}, Passed: {stats['passed']}, "
                  f"Failed: {stats['failed']}, Errors: {stats['errors']}, "
                  f"Skipped: {stats['skipped']}, Timeouts: {stats['timeouts']}, "
                  f"Success Rate: {category_success_rate:.1f}%")
        
        print(f"\n🎯 PRIORITY BREAKDOWN:")
        for priority, stats in report['priority_stats'].items():
            priority_success_rate = (stats['passed'] / stats['tests'] * 100) if stats['tests'] > 0 else 0
            print(f"  {priority.upper()}:")
            print(f"    Tests: {stats['tests']}, Passed: {stats['passed']}, "
                  f"Failed: {stats['failed']}, Errors: {stats['errors']}, "
                  f"Skipped: {stats['skipped']}, Timeouts: {stats['timeouts']}, "
                  f"Success Rate: {priority_success_rate:.1f}%")
        
        print(f"\n🏷️  TAG BREAKDOWN:")
        for tag, stats in report['tag_stats'].items():
            tag_success_rate = (stats['passed'] / stats['tests'] * 100) if stats['tests'] > 0 else 0
            print(f"  #{tag}:")
            print(f"    Tests: {stats['tests']}, Passed: {stats['passed']}, "
                  f"Failed: {stats['failed']}, Errors: {stats['errors']}, "
                  f"Skipped: {stats['skipped']}, Timeouts: {stats['timeouts']}, "
                  f"Success Rate: {tag_success_rate:.1f}%")
        
        print(f"\n🔬 OPTIMIZATION BREAKDOWN:")
        for opt_type, stats in report['optimization_stats'].items():
            opt_success_rate = (stats['passed'] / stats['tests'] * 100) if stats['tests'] > 0 else 0
            print(f"  {opt_type.upper()}:")
            print(f"    Tests: {stats['tests']}, Passed: {stats['passed']}, "
                  f"Failed: {stats['failed']}, Errors: {stats['errors']}, "
                  f"Success Rate: {opt_success_rate:.1f}%")
        
        print(f"\n💎 QUALITY METRICS:")
        for category, stats in report['quality_stats'].items():
            print(f"  {category.upper()}:")
            print(f"    Average Quality: {stats['avg']:.3f}")
            print(f"    Min Quality: {stats['min']:.3f}")
            print(f"    Max Quality: {stats['max']:.3f}")
            print(f"    Count: {stats['count']}")
        
        print(f"\n🛡️  RELIABILITY METRICS:")
        for category, stats in report['reliability_stats'].items():
            print(f"  {category.upper()}:")
            print(f"    Average Reliability: {stats['avg']:.3f}")
            print(f"    Min Reliability: {stats['min']:.3f}")
            print(f"    Max Reliability: {stats['max']:.3f}")
            print(f"    Count: {stats['count']}")
        
        print(f"\n⚡ PERFORMANCE METRICS:")
        for category, stats in report['performance_stats'].items():
            print(f"  {category.upper()}:")
            print(f"    Average Performance: {stats['avg']:.3f}")
            print(f"    Min Performance: {stats['min']:.3f}")
            print(f"    Max Performance: {stats['max']:.3f}")
            print(f"    Count: {stats['count']}")
        
        print(f"\n🔧 EFFICIENCY METRICS:")
        for category, stats in report['efficiency_stats'].items():
            print(f"  {category.upper()}:")
            print(f"    Average Efficiency: {stats['avg']:.3f}")
            print(f"    Min Efficiency: {stats['min']:.3f}")
            print(f"    Max Efficiency: {stats['max']:.3f}")
            print(f"    Count: {stats['count']}")
        
        print(f"\n📈 SCALABILITY METRICS:")
        for category, stats in report['scalability_stats'].items():
            print(f"  {category.upper()}:")
            print(f"    Average Scalability: {stats['avg']:.3f}")
            print(f"    Min Scalability: {stats['min']:.3f}")
            print(f"    Max Scalability: {stats['max']:.3f}")
            print(f"    Count: {stats['count']}")
        
        print(f"\n💻 SYSTEM INFORMATION:")
        system_info = report['system_info']
        print(f"  Python Version: {system_info['python_version']}")
        print(f"  Platform: {system_info['platform']}")
        print(f"  CPU Count: {system_info['cpu_count']}")
        print(f"  Memory: {system_info['memory_gb']:.1f}GB")
        print(f"  Execution Mode: {system_info['execution_mode']}")
        print(f"  Max Workers: {system_info['max_workers']}")
        
        # Print recommendations if available
        if 'recommendations' in report and report['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, recommendation in enumerate(report['recommendations'], 1):
                print(f"  {i}. {recommendation}")
        
        # Print failures and errors
        if summary['failed'] > 0 or summary['errors'] > 0:
            print(f"\n❌ FAILURES AND ERRORS:")
            for result in report['detailed_results']:
                if result['success'] and result['result']:
                    test_result = result['result']
                    if test_result.failures:
                        print(f"\n  Failures in {result['suite_name']}:")
                        for test, traceback in test_result.failures:
                            print(f"    - {test}: {traceback}")
                    
                    if test_result.errors:
                        print(f"\n  Errors in {result['suite_name']}:")
                        for test, traceback in test_result.errors:
                            print(f"    - {test}: {traceback}")
        
        print("\n" + "="*140)
    
    def save_comprehensive_report(self, report: Dict[str, Any], output_file: Optional[str] = None):
        """Save comprehensive test report to file."""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"test_report_{timestamp}.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"📄 Comprehensive test report saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error saving report: {e}")
    
    def generate_html_report(self, report: Dict[str, Any], output_file: str):
        """Generate HTML report."""
        try:
            html_content = self._generate_html_content(report)
            
            with open(output_file, 'w') as f:
                f.write(html_content)
            
            self.logger.info(f"📄 HTML report generated: {output_file}")
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {e}")
    
    def _generate_html_content(self, report: Dict[str, Any]) -> str:
        """Generate HTML content for the report."""
        summary = report['summary']
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Optimization Core Test Report V4</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 8px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #007bff;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        .section h2 {{
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }}
        .metric-card h4 {{
            margin: 0 0 10px 0;
            color: #495057;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #28a745;
        }}
        .success-rate {{
            color: {self._get_success_rate_color(summary['success_rate'])};
        }}
        .recommendations {{
            background: #e3f2fd;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #2196f3;
        }}
        .recommendations h3 {{
            margin-top: 0;
            color: #1976d2;
        }}
        .recommendations ul {{
            margin: 0;
            padding-left: 20px;
        }}
        .recommendations li {{
            margin-bottom: 8px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Enhanced Optimization Core Test Report V4</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="summary">
            <div class="summary-card">
                <h3>Total Tests</h3>
                <div class="value">{summary['total_tests']}</div>
            </div>
            <div class="summary-card">
                <h3>Passed</h3>
                <div class="value success-rate">{summary['passed']}</div>
            </div>
            <div class="summary-card">
                <h3>Failed</h3>
                <div class="value">{summary['failed']}</div>
            </div>
            <div class="summary-card">
                <h3>Success Rate</h3>
                <div class="value success-rate">{summary['success_rate']:.1f}%</div>
            </div>
            <div class="summary-card">
                <h3>Execution Time</h3>
                <div class="value">{summary['total_execution_time']:.2f}s</div>
            </div>
            <div class="summary-card">
                <h3>Memory Usage</h3>
                <div class="value">{summary['total_memory_usage']:.2f}MB</div>
            </div>
        </div>
        
        <div class="section">
            <h2>📈 Category Breakdown</h2>
            <div class="metrics-grid">
        """
        
        for category, stats in report['category_stats'].items():
            category_success_rate = (stats['passed'] / stats['tests'] * 100) if stats['tests'] > 0 else 0
            html += f"""
                <div class="metric-card">
                    <h4>{category.upper()}</h4>
                    <p>Tests: {stats['tests']}</p>
                    <p>Passed: {stats['passed']}</p>
                    <p>Failed: {stats['failed']}</p>
                    <p>Errors: {stats['errors']}</p>
                    <p>Success Rate: <span class="success-rate">{category_success_rate:.1f}%</span></p>
                </div>
            """
        
        html += """
            </div>
        </div>
        
        <div class="section">
            <h2>🎯 Priority Breakdown</h2>
            <div class="metrics-grid">
        """
        
        for priority, stats in report['priority_stats'].items():
            priority_success_rate = (stats['passed'] / stats['tests'] * 100) if stats['tests'] > 0 else 0
            html += f"""
                <div class="metric-card">
                    <h4>{priority.upper()}</h4>
                    <p>Tests: {stats['tests']}</p>
                    <p>Passed: {stats['passed']}</p>
                    <p>Failed: {stats['failed']}</p>
                    <p>Errors: {stats['errors']}</p>
                    <p>Success Rate: <span class="success-rate">{priority_success_rate:.1f}%</span></p>
                </div>
            """
        
        html += """
            </div>
        </div>
        
        <div class="section">
            <h2>💎 Quality Metrics</h2>
            <div class="metrics-grid">
        """
        
        for category, stats in report['quality_stats'].items():
            html += f"""
                <div class="metric-card">
                    <h4>{category.upper()}</h4>
                    <p>Average Quality: <span class="metric-value">{stats['avg']:.3f}</span></p>
                    <p>Min Quality: {stats['min']:.3f}</p>
                    <p>Max Quality: {stats['max']:.3f}</p>
                    <p>Count: {stats['count']}</p>
                </div>
            """
        
        html += """
            </div>
        </div>
        
        <div class="section">
            <h2>🛡️ Reliability Metrics</h2>
            <div class="metrics-grid">
        """
        
        for category, stats in report['reliability_stats'].items():
            html += f"""
                <div class="metric-card">
                    <h4>{category.upper()}</h4>
                    <p>Average Reliability: <span class="metric-value">{stats['avg']:.3f}</span></p>
                    <p>Min Reliability: {stats['min']:.3f}</p>
                    <p>Max Reliability: {stats['max']:.3f}</p>
                    <p>Count: {stats['count']}</p>
                </div>
            """
        
        html += """
            </div>
        </div>
        
        <div class="section">
            <h2>⚡ Performance Metrics</h2>
            <div class="metrics-grid">
        """
        
        for category, stats in report['performance_stats'].items():
            html += f"""
                <div class="metric-card">
                    <h4>{category.upper()}</h4>
                    <p>Average Performance: <span class="metric-value">{stats['avg']:.3f}</span></p>
                    <p>Min Performance: {stats['min']:.3f}</p>
                    <p>Max Performance: {stats['max']:.3f}</p>
                    <p>Count: {stats['count']}</p>
                </div>
            """
        
        html += """
            </div>
        </div>
        
        <div class="section">
            <h2>🔧 Efficiency Metrics</h2>
            <div class="metrics-grid">
        """
        
        for category, stats in report['efficiency_stats'].items():
            html += f"""
                <div class="metric-card">
                    <h4>{category.upper()}</h4>
                    <p>Average Efficiency: <span class="metric-value">{stats['avg']:.3f}</span></p>
                    <p>Min Efficiency: {stats['min']:.3f}</p>
                    <p>Max Efficiency: {stats['max']:.3f}</p>
                    <p>Count: {stats['count']}</p>
                </div>
            """
        
        html += """
            </div>
        </div>
        
        <div class="section">
            <h2>📈 Scalability Metrics</h2>
            <div class="metrics-grid">
        """
        
        for category, stats in report['scalability_stats'].items():
            html += f"""
                <div class="metric-card">
                    <h4>{category.upper()}</h4>
                    <p>Average Scalability: <span class="metric-value">{stats['avg']:.3f}</span></p>
                    <p>Min Scalability: {stats['min']:.3f}</p>
                    <p>Max Scalability: {stats['max']:.3f}</p>
                    <p>Count: {stats['count']}</p>
                </div>
            """
        
        html += """
            </div>
        </div>
        """
        
        # Add recommendations if available
        if 'recommendations' in report and report['recommendations']:
            html += """
        <div class="section">
            <div class="recommendations">
                <h3>💡 Recommendations</h3>
                <ul>
            """
            for recommendation in report['recommendations']:
                html += f"<li>{recommendation}</li>"
            
            html += """
                </ul>
            </div>
        </div>
            """
        
        html += """
    </div>
</body>
</html>
        """
        
        return html
    
    def _get_success_rate_color(self, success_rate: float) -> str:
        """Get color for success rate."""
        if success_rate >= 95:
            return "#28a745"  # Green
        elif success_rate >= 80:
            return "#ffc107"  # Yellow
        else:
            return "#dc3545"  # Red
    
    def generate_csv_report(self, report: Dict[str, Any], output_file: str):
        """Generate CSV report."""
        try:
            import csv
            
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write summary
                writer.writerow(['Metric', 'Value'])
                writer.writerow(['Total Tests', report['summary']['total_tests']])
                writer.writerow(['Passed', report['summary']['passed']])
                writer.writerow(['Failed', report['summary']['failed']])
                writer.writerow(['Errors', report['summary']['errors']])
                writer.writerow(['Success Rate', f"{report['summary']['success_rate']:.1f}%"])
                writer.writerow(['Execution Time', f"{report['summary']['total_execution_time']:.2f}s"])
                writer.writerow(['Memory Usage', f"{report['summary']['total_memory_usage']:.2f}MB"])
                
                # Write category stats
                writer.writerow([])
                writer.writerow(['Category', 'Tests', 'Passed', 'Failed', 'Errors', 'Success Rate'])
                for category, stats in report['category_stats'].items():
                    success_rate = (stats['passed'] / stats['tests'] * 100) if stats['tests'] > 0 else 0
                    writer.writerow([
                        category, stats['tests'], stats['passed'], 
                        stats['failed'], stats['errors'], f"{success_rate:.1f}%"
                    ])
            
            self.logger.info(f"📄 CSV report generated: {output_file}")
        except Exception as e:
            self.logger.error(f"Error generating CSV report: {e}")
    
    def generate_markdown_report(self, report: Dict[str, Any], output_file: str):
        """Generate Markdown report."""
        try:
            summary = report['summary']
            
            markdown = f"""# Enhanced Optimization Core Test Report V4

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Overall Summary

| Metric | Value |
|--------|-------|
| Total Tests | {summary['total_tests']} |
| Passed | {summary['passed']} |
| Failed | {summary['failed']} |
| Errors | {summary['errors']} |
| Skipped | {summary['skipped']} |
| Timeouts | {summary['timeouts']} |
| Success Rate | {summary['success_rate']:.1f}% |
| Execution Time | {summary['total_execution_time']:.2f}s |
| Memory Usage | {summary['total_memory_usage']:.2f}MB |

## 📈 Category Breakdown

| Category | Tests | Passed | Failed | Errors | Success Rate |
|----------|-------|--------|--------|--------|--------------|
"""
            
            for category, stats in report['category_stats'].items():
                success_rate = (stats['passed'] / stats['tests'] * 100) if stats['tests'] > 0 else 0
                markdown += f"| {category} | {stats['tests']} | {stats['passed']} | {stats['failed']} | {stats['errors']} | {success_rate:.1f}% |\n"
            
            markdown += "\n## 🎯 Priority Breakdown\n\n"
            markdown += "| Priority | Tests | Passed | Failed | Errors | Success Rate |\n"
            markdown += "|----------|-------|--------|--------|--------|--------------|\n"
            
            for priority, stats in report['priority_stats'].items():
                success_rate = (stats['passed'] / stats['tests'] * 100) if stats['tests'] > 0 else 0
                markdown += f"| {priority} | {stats['tests']} | {stats['passed']} | {stats['failed']} | {stats['errors']} | {success_rate:.1f}% |\n"
            
            markdown += "\n## 💎 Quality Metrics\n\n"
            markdown += "| Category | Average | Min | Max | Count |\n"
            markdown += "|----------|---------|-----|-----|-------|\n"
            
            for category, stats in report['quality_stats'].items():
                markdown += f"| {category} | {stats['avg']:.3f} | {stats['min']:.3f} | {stats['max']:.3f} | {stats['count']} |\n"
            
            markdown += "\n## 🛡️ Reliability Metrics\n\n"
            markdown += "| Category | Average | Min | Max | Count |\n"
            markdown += "|----------|---------|-----|-----|-------|\n"
            
            for category, stats in report['reliability_stats'].items():
                markdown += f"| {category} | {stats['avg']:.3f} | {stats['min']:.3f} | {stats['max']:.3f} | {stats['count']} |\n"
            
            markdown += "\n## ⚡ Performance Metrics\n\n"
            markdown += "| Category | Average | Min | Max | Count |\n"
            markdown += "|----------|---------|-----|-----|-------|\n"
            
            for category, stats in report['performance_stats'].items():
                markdown += f"| {category} | {stats['avg']:.3f} | {stats['min']:.3f} | {stats['max']:.3f} | {stats['count']} |\n"
            
            markdown += "\n## 🔧 Efficiency Metrics\n\n"
            markdown += "| Category | Average | Min | Max | Count |\n"
            markdown += "|----------|---------|-----|-----|-------|\n"
            
            for category, stats in report['efficiency_stats'].items():
                markdown += f"| {category} | {stats['avg']:.3f} | {stats['min']:.3f} | {stats['max']:.3f} | {stats['count']} |\n"
            
            markdown += "\n## 📈 Scalability Metrics\n\n"
            markdown += "| Category | Average | Min | Max | Count |\n"
            markdown += "|----------|---------|-----|-----|-------|\n"
            
            for category, stats in report['scalability_stats'].items():
                markdown += f"| {category} | {stats['avg']:.3f} | {stats['min']:.3f} | {stats['max']:.3f} | {stats['count']} |\n"
            
            # Add recommendations if available
            if 'recommendations' in report and report['recommendations']:
                markdown += "\n## 💡 Recommendations\n\n"
                for i, recommendation in enumerate(report['recommendations'], 1):
                    markdown += f"{i}. {recommendation}\n"
            
            with open(output_file, 'w') as f:
                f.write(markdown)
            
            self.logger.info(f"📄 Markdown report generated: {output_file}")
        except Exception as e:
            self.logger.error(f"Error generating Markdown report: {e}")











