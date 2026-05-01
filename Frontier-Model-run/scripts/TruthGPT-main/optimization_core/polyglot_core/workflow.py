"""
Workflow orchestration for polyglot_core.

Provides workflow definition and execution.
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class StepStatus(str, Enum):
    """Step status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """Workflow step definition."""
    id: str
    name: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    retry_on_failure: bool = False
    max_retries: int = 0
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class Workflow:
    """
    Workflow orchestrator for polyglot_core.
    
    Executes steps in dependency order with error handling.
    """
    
    def __init__(self, name: str):
        """
        Initialize workflow.
        
        Args:
            name: Workflow name
        """
        self.name = name
        self.steps: Dict[str, WorkflowStep] = {}
        self.context: Dict[str, Any] = {}
        self._execution_order: List[str] = []
    
    def add_step(
        self,
        step_id: str,
        name: str,
        func: Callable,
        *args,
        depends_on: Optional[List[str]] = None,
        condition: Optional[Callable[[Dict[str, Any]], bool]] = None,
        retry_on_failure: bool = False,
        max_retries: int = 0,
        **kwargs
    ):
        """
        Add step to workflow.
        
        Args:
            step_id: Unique step ID
            name: Step name
            func: Function to execute
            *args: Positional arguments
            depends_on: List of step IDs this step depends on
            condition: Optional condition function
            retry_on_failure: Whether to retry on failure
            max_retries: Maximum retry attempts
            **kwargs: Keyword arguments
        """
        step = WorkflowStep(
            id=step_id,
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
            depends_on=depends_on or [],
            condition=condition,
            retry_on_failure=retry_on_failure,
            max_retries=max_retries
        )
        
        self.steps[step_id] = step
    
    def _compute_execution_order(self) -> List[str]:
        """Compute execution order based on dependencies."""
        # Topological sort
        in_degree = {step_id: len(step.depends_on) for step_id, step in self.steps.items()}
        queue = [step_id for step_id, degree in in_degree.items() if degree == 0]
        order = []
        
        while queue:
            step_id = queue.pop(0)
            order.append(step_id)
            
            for other_id, other_step in self.steps.items():
                if step_id in other_step.depends_on:
                    in_degree[other_id] -= 1
                    if in_degree[other_id] == 0:
                        queue.append(other_id)
        
        if len(order) != len(self.steps):
            raise ValueError("Circular dependency detected in workflow")
        
        return order
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute workflow.
        
        Returns:
            Dictionary with step results
        """
        self._execution_order = self._compute_execution_order()
        results = {}
        
        for step_id in self._execution_order:
            step = self.steps[step_id]
            
            # Check condition
            if step.condition and not step.condition(self.context):
                step.status = StepStatus.SKIPPED
                continue
            
            # Check dependencies
            if any(self.steps[dep_id].status != StepStatus.COMPLETED for dep_id in step.depends_on):
                step.status = StepStatus.FAILED
                step.error = "Dependencies not completed"
                continue
            
            # Execute step
            step.status = StepStatus.RUNNING
            step.started_at = datetime.now()
            
            retries = 0
            while retries <= step.max_retries:
                try:
                    result = step.func(*step.args, **step.kwargs, **self.context)
                    step.result = result
                    step.status = StepStatus.COMPLETED
                    step.completed_at = datetime.now()
                    self.context[step_id] = result
                    results[step_id] = result
                    break
                except Exception as e:
                    if retries < step.max_retries and step.retry_on_failure:
                        retries += 1
                        continue
                    else:
                        step.error = str(e)
                        step.status = StepStatus.FAILED
                        step.completed_at = datetime.now()
                        break
        
        return results
    
    def get_status(self) -> Dict[str, str]:
        """Get status of all steps."""
        return {step_id: step.status.value for step_id, step in self.steps.items()}


def create_workflow(name: str) -> Workflow:
    """Convenience function to create workflow."""
    return Workflow(name)













