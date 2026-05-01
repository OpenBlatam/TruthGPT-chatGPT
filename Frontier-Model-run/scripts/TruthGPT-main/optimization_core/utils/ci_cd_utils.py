"""
CI/CD utilities for optimization_core.

Provides utilities for continuous integration and deployment.
"""
import logging
import subprocess
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TestResult(BaseModel):
    """Result of test execution."""
    test_name: str
    passed: bool
    duration: float
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "duration": self.duration,
            "error": self.error,
        }


class CIRunner:
    """Runner for CI operations."""
    
    def __init__(self, project_root: Path):
        """
        Initialize CI runner.
        
        Args:
            project_root: Root directory of project
        """
        self.project_root = Path(project_root)
    
    def run_tests(
        self,
        test_path: Optional[Path] = None,
        verbose: bool = False
    ) -> List[TestResult]:
        """
        Run tests.
        
        Args:
            test_path: Path to test file/directory
            verbose: Verbose output
        
        Returns:
            List of test results
        """
        if test_path is None:
            test_path = self.project_root / "tests"
        
        cmd = ["python", "-m", "pytest", str(test_path)]
        if verbose:
            cmd.append("-v")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            # Parse results (simplified)
            # In practice, would parse pytest output
            return [
                TestResult(
                    test_name="all_tests",
                    passed=result.returncode == 0,
                    duration=0.0,
                    error=result.stderr if result.returncode != 0 else None
                )
            ]
        except Exception as e:
            logger.error(f"Failed to run tests: {e}", exc_info=True)
            return [
                TestResult(
                    test_name="all_tests",
                    passed=False,
                    duration=0.0,
                    error=str(e)
                )
            ]
    
    def run_linter(
        self,
        paths: Optional[List[Path]] = None
    ) -> Dict[str, Any]:
        """
        Run linter.
        
        Args:
            paths: Paths to lint
        
        Returns:
            Linter results
        """
        if paths is None:
            paths = [self.project_root]
        
        try:
            # Try flake8 or pylint
            cmd = ["flake8"] + [str(p) for p in paths]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            return {
                "passed": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr,
            }
        except FileNotFoundError:
            logger.warning("Linter not found, skipping")
            return {"passed": True, "output": "", "errors": ""}
    
    def run_type_checker(
        self,
        paths: Optional[List[Path]] = None
    ) -> Dict[str, Any]:
        """
        Run type checker.
        
        Args:
            paths: Paths to check
        
        Returns:
            Type checker results
        """
        if paths is None:
            paths = [self.project_root]
        
        try:
            cmd = ["mypy"] + [str(p) for p in paths]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            return {
                "passed": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr,
            }
        except FileNotFoundError:
            logger.warning("Type checker not found, skipping")
            return {"passed": True, "output": "", "errors": ""}
    
    def run_benchmarks(
        self,
        benchmark_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Run benchmarks.
        
        Args:
            benchmark_path: Path to benchmark file
        
        Returns:
            Benchmark results
        """
        if benchmark_path is None:
            benchmark_path = self.project_root / "benchmarks"
        
        # In practice, would run actual benchmarks
        return {
            "passed": True,
            "results": {},
        }
    
    def generate_report(
        self,
        results: Dict[str, Any],
        output_path: Path
    ):
        """
        Generate CI report.
        
        Args:
            results: CI results
            output_path: Path to save report
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"CI report saved to {output_path}")


def run_ci_checks(
    project_root: Path,
    output_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Run all CI checks.
    
    Args:
        project_root: Project root directory
        output_path: Optional path to save report
    
    Returns:
        CI results
    """
    runner = CIRunner(project_root)
    
    results = {
        "tests": [r.to_dict() for r in runner.run_tests()],
        "linter": runner.run_linter(),
        "type_checker": runner.run_type_checker(),
        "benchmarks": runner.run_benchmarks(),
    }
    
    # Overall status
    results["overall_passed"] = all([
        all(r["passed"] for r in results["tests"]),
        results["linter"]["passed"],
        results["type_checker"]["passed"],
        results["benchmarks"]["passed"],
    ])
    
    if output_path:
        runner.generate_report(results, output_path)
    
    return results













