"""
Integration testing utilities for optimization_core.

Provides utilities for integration testing.
"""
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from pydantic import BaseModel, Field
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class IntegrationTestResult(BaseModel):
    """Result of integration test."""
    test_name: str
    passed: bool
    duration: float
    components_tested: List[str]
    errors: List[str] = Field(default_factory=list)


class IntegrationTestRunner:
    """Runner for integration tests."""
    
    def __init__(self):
        """Initialize integration test runner."""
        self.components: Dict[str, Any] = {}
        self.results: List[IntegrationTestResult] = []
    
    def register_component(
        self,
        name: str,
        component: Any
    ):
        """
        Register a component for testing.
        
        Args:
            name: Component name
            component: Component instance
        """
        self.components[name] = component
        logger.debug(f"Registered component: {name}")
    
    def run_integration_test(
        self,
        test_name: str,
        test_func: Callable,
        components: List[str]
    ) -> IntegrationTestResult:
        """
        Run an integration test.
        
        Args:
            test_name: Test name
            test_func: Test function
            components: List of component names to test
        
        Returns:
            Test result
        """
        # Verify components exist
        missing = [c for c in components if c not in self.components]
        if missing:
            return IntegrationTestResult(
                test_name=test_name,
                passed=False,
                duration=0.0,
                components_tested=components,
                errors=[f"Missing components: {missing}"]
            )
        
        # Get component instances
        component_instances = {c: self.components[c] for c in components}
        
        # Run test
        start_time = time.time()
        errors = []
        
        try:
            test_func(**component_instances)
            passed = True
        except Exception as e:
            passed = False
            errors.append(str(e))
            logger.error(f"Integration test '{test_name}' failed: {e}", exc_info=True)
        
        duration = time.time() - start_time
        
        result = IntegrationTestResult(
            test_name=test_name,
            passed=passed,
            duration=duration,
            components_tested=components,
            errors=errors
        )
        
        self.results.append(result)
        return result
    
    @contextmanager
    def test_context(self, components: List[str]):
        """
        Context manager for integration testing.
        
        Args:
            components: List of component names
        
        Yields:
            Component instances
        """
        # Verify components exist
        missing = [c for c in components if c not in self.components]
        if missing:
            raise ValueError(f"Missing components: {missing}")
        
        # Get component instances
        component_instances = {c: self.components[c] for c in components}
        
        try:
            yield component_instances
        finally:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get test statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.results:
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "avg_duration": 0.0,
            }
        
        return {
            "total": len(self.results),
            "passed": len([r for r in self.results if r.passed]),
            "failed": len([r for r in self.results if not r.passed]),
            "avg_duration": sum(r.duration for r in self.results) / len(self.results),
        }


def create_integration_test_runner() -> IntegrationTestRunner:
    """
    Create an integration test runner.
    
    Returns:
        Integration test runner
    """
    return IntegrationTestRunner()













