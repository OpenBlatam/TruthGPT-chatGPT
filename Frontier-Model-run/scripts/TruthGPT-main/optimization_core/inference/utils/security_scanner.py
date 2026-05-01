"""
🔒 Security Scanner
Comprehensive security scanning and validation for inference API
"""

import os
import re
import subprocess
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


@dataclass
class SecurityIssue:
    """Security issue found"""
    severity: str  # critical, high, medium, low
    category: str
    description: str
    recommendation: str
    location: Optional[str] = None


class SecurityScanner:
    """Security scanner for inference API"""
    
    def __init__(self, api_url: str = "http://localhost:8080"):
        self.api_url = api_url
        self.issues: List[SecurityIssue] = []
    
    def scan_config(self) -> List[SecurityIssue]:
        """Scan configuration for security issues"""
        issues = []
        
        # Check for default tokens
        api_token = os.getenv("TRUTHGPT_API_TOKEN", "")
        if api_token in ["changeme", "default", ""]:
            issues.append(SecurityIssue(
                severity="critical",
                category="Authentication",
                description="Default or empty API token detected",
                recommendation="Set a strong, unique API token in TRUTHGPT_API_TOKEN"
            ))
        
        # Check webhook secret
        webhook_secret = os.getenv("WEBHOOK_HMAC_SECRET", "")
        if webhook_secret in ["changeme-secret", "default", ""]:
            issues.append(SecurityIssue(
                severity="critical",
                category="Webhooks",
                description="Default or empty webhook HMAC secret",
                recommendation="Set a strong webhook secret in WEBHOOK_HMAC_SECRET"
            ))
        
        # Check CORS
        cors_origins = os.getenv("CORS_ORIGINS", "*")
        if cors_origins == "*":
            issues.append(SecurityIssue(
                severity="high",
                category="CORS",
                description="CORS allows all origins (*)",
                recommendation="Restrict CORS_ORIGINS to specific domains"
            ))
        
        # Check Redis URL
        redis_url = os.getenv("REDIS_URL", "")
        if redis_url and "localhost" in redis_url and "password" not in redis_url:
            issues.append(SecurityIssue(
                severity="medium",
                category="Database",
                description="Redis connection without password",
                recommendation="Use Redis with authentication enabled"
            ))
        
        return issues
    
    def scan_api_endpoints(self) -> List[SecurityIssue]:
        """Scan API endpoints for security issues"""
        issues = []
        
        if not HTTPX_AVAILABLE:
            return issues
        
        try:
            client = httpx.Client(timeout=5.0)
            
            # Check if health endpoint exposes sensitive info
            try:
                response = client.get(f"{self.api_url}/health")
                if response.status_code == 200:
                    data = response.json()
                    # Check for sensitive information
                    if "secret" in json.dumps(data).lower() or "token" in json.dumps(data).lower():
                        issues.append(SecurityIssue(
                            severity="high",
                            category="Information Disclosure",
                            description="Health endpoint may expose sensitive information",
                            recommendation="Review health endpoint response for sensitive data"
                        ))
            except:
                pass
            
            # Check metrics endpoint protection
            try:
                response = client.get(f"{self.api_url}/metrics")
                if response.status_code == 200:
                    # Metrics should ideally be protected
                    issues.append(SecurityIssue(
                        severity="medium",
                        category="Access Control",
                        description="Metrics endpoint is publicly accessible",
                        recommendation="Consider adding authentication or IP whitelisting for /metrics"
                    ))
            except:
                pass
            
            # Check for missing security headers
            try:
                response = client.get(f"{self.api_url}/")
                headers = response.headers
                
                security_headers = {
                    "X-Content-Type-Options": "nosniff",
                    "X-Frame-Options": "DENY",
                    "X-XSS-Protection": "1; mode=block",
                    "Strict-Transport-Security": "max-age=31536000"
                }
                
                for header, value in security_headers.items():
                    if header not in headers:
                        issues.append(SecurityIssue(
                            severity="medium",
                            category="Headers",
                            description=f"Missing security header: {header}",
                            recommendation=f"Add {header}: {value} header to responses"
                        ))
            except:
                pass
            
        except Exception:
            # API not available, skip
            pass
        
        return issues
    
    def scan_dependencies(self) -> List[SecurityIssue]:
        """Scan dependencies for known vulnerabilities"""
        issues = []
        
        # Check if pip-audit is available
        try:
            result = subprocess.run(
                ["pip-audit", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                try:
                    vulns = json.loads(result.stdout)
                    for vuln in vulns.get("vulnerabilities", []):
                        severity = vuln.get("severity", "unknown").lower()
                        if severity in ["critical", "high"]:
                            issues.append(SecurityIssue(
                                severity=severity,
                                category="Dependencies",
                                description=f"Vulnerability in {vuln.get('name', 'unknown')}: {vuln.get('description', '')}",
                                recommendation=f"Update {vuln.get('name')} to version {vuln.get('fix_versions', ['latest'])[0]}"
                            ))
                except:
                    pass
        except FileNotFoundError:
            issues.append(SecurityIssue(
                severity="low",
                category="Tools",
                description="pip-audit not available for dependency scanning",
                recommendation="Install pip-audit: pip install pip-audit"
            ))
        except Exception:
            pass
        
        return issues
    
    def scan_files(self, directory: str = ".") -> List[SecurityIssue]:
        """Scan code files for security issues"""
        issues = []
        
        patterns = {
            r"password\s*=\s*['\"][^'\"]+['\"]": {
                "severity": "critical",
                "category": "Secrets",
                "description": "Hardcoded password detected",
                "recommendation": "Use environment variables or secrets management"
            },
            r"api[_-]?key\s*=\s*['\"][^'\"]+['\"]": {
                "severity": "critical",
                "category": "Secrets",
                "description": "Hardcoded API key detected",
                "recommendation": "Use environment variables or secrets management"
            },
            r"secret\s*=\s*['\"][^'\"]+['\"]": {
                "severity": "high",
                "category": "Secrets",
                "description": "Hardcoded secret detected",
                "recommendation": "Use environment variables or secrets management"
            },
            r"eval\(": {
                "severity": "critical",
                "category": "Code Injection",
                "description": "Use of eval() detected",
                "recommendation": "Avoid eval() - use safe alternatives"
            },
            r"exec\(": {
                "severity": "critical",
                "category": "Code Injection",
                "description": "Use of exec() detected",
                "recommendation": "Avoid exec() - use safe alternatives"
            },
            r"subprocess\.call\([^)]*shell\s*=\s*True": {
                "severity": "high",
                "category": "Command Injection",
                "description": "Shell execution with user input detected",
                "recommendation": "Avoid shell=True, use explicit command lists"
            }
        }
        
        for file_path in Path(directory).rglob("*.py"):
            if "venv" in str(file_path) or "__pycache__" in str(file_path):
                continue
            
            try:
                content = file_path.read_text()
                for pattern, issue_info in patterns.items():
                    if re.search(pattern, content, re.IGNORECASE):
                        issues.append(SecurityIssue(
                            severity=issue_info["severity"],
                            category=issue_info["category"],
                            description=f"{issue_info['description']} in {file_path}",
                            recommendation=issue_info["recommendation"],
                            location=str(file_path)
                        ))
            except Exception:
                continue
        
        return issues
    
    def run_full_scan(self) -> List[SecurityIssue]:
        """Run complete security scan"""
        print("🔒 Running security scan...")
        
        all_issues = []
        all_issues.extend(self.scan_config())
        all_issues.extend(self.scan_api_endpoints())
        all_issues.extend(self.scan_dependencies())
        all_issues.extend(self.scan_files())
        
        self.issues = all_issues
        return all_issues
    
    def print_report(self):
        """Print security scan report"""
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        
        console = Console()
        
        if not self.issues:
            console.print("[green]✓ No security issues found![/green]")
            return
        
        # Group by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_issues = sorted(self.issues, key=lambda x: severity_order.get(x.severity, 4))
        
        # Create table
        table = Table(title="🔒 Security Issues Found")
        table.add_column("Severity", style="red")
        table.add_column("Category", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Recommendation", style="yellow")
        
        for issue in sorted_issues:
            severity_color = {
                "critical": "red",
                "high": "yellow",
                "medium": "blue",
                "low": "green"
            }.get(issue.severity, "white")
            
            table.add_row(
                f"[{severity_color}]{issue.severity.upper()}[/{severity_color}]",
                issue.category,
                issue.description,
                issue.recommendation
            )
        
        console.print(table)
        
        # Summary
        counts = {}
        for issue in self.issues:
            counts[issue.severity] = counts.get(issue.severity, 0) + 1
        
        summary = f"Total Issues: {len(self.issues)}\n"
        for severity in ["critical", "high", "medium", "low"]:
            if severity in counts:
                summary += f"{severity.capitalize()}: {counts[severity]}\n"
        
        console.print(Panel(summary, title="Summary", border_style="yellow"))


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Security Scanner")
    parser.add_argument("--url", default="http://localhost:8080", help="API URL")
    parser.add_argument("--directory", default=".", help="Directory to scan")
    
    args = parser.parse_args()
    
    scanner = SecurityScanner(args.url)
    scanner.run_full_scan()
    scanner.print_report()


if __name__ == "__main__":
    main()



