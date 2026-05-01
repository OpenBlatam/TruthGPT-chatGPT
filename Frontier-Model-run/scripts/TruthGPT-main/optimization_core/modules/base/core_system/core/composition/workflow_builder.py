"""
Workflow builder for creating training/inference workflows declaratively.
"""
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field

from ..service_registry import ServiceContainer

logger = logging.getLogger(__name__)


@dataclass
class WorkflowStep:
    """Represents a step in a workflow."""
    name: str
    step_type: str
    component: str
    config: Dict[str, Any] = field(default_factory=dict)
    condition: Optional[Callable] = None
    on_success: Optional[str] = None
    on_error: Optional[str] = None


class WorkflowBuilder:
    """
    Builder for creating declarative workflows.
    Enables defining complex workflows in a simple, modular way.
    """
    
    def __init__(self, container: Optional[ServiceContainer] = None):
        """
        Initialize workflow builder.
        
        Args:
            container: Service container
        """
        self.container = container or ServiceContainer()
        self.steps: List[WorkflowStep] = []
        self.current_step: int = 0
    
    def add_step(
        self,
        name: str,
        step_type: str,
        component: str,
        config: Optional[Dict[str, Any]] = None,
        condition: Optional[Callable] = None,
        on_success: Optional[str] = None,
        on_error: Optional[str] = None,
    ) -> "WorkflowBuilder":
        """
        Add a step to the workflow.
        
        Args:
            name: Step name
            step_type: Type of step (load_model|train|evaluate|save)
            component: Component to use
            config: Step configuration
            condition: Optional condition function
            on_success: Next step on success
            on_error: Next step on error
        
        Returns:
            Self for chaining
        """
        step = WorkflowStep(
            name=name,
            step_type=step_type,
            component=component,
            config=config or {},
            condition=condition,
            on_success=on_success,
            on_error=on_error,
        )
        self.steps.append(step)
        return self
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute the workflow.
        
        Returns:
            Dictionary with execution results
        """
        results = {}
        self.current_step = 0
        
        logger.info(f"Executing workflow with {len(self.steps)} steps")
        
        while self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            
            # Check condition
            if step.condition and not step.condition(results):
                logger.debug(f"Skipping step '{step.name}' (condition not met)")
                self.current_step += 1
                continue
            
            try:
                # Execute step
                result = self._execute_step(step, results)
                results[step.name] = result
                
                logger.info(f"Step '{step.name}' completed successfully")
                
                # Move to next step or conditional step
                if step.on_success:
                    next_step = self._find_step(step.on_success)
                    if next_step is not None:
                        self.current_step = next_step
                    else:
                        self.current_step += 1
                else:
                    self.current_step += 1
                    
            except Exception as e:
                logger.error(f"Error in step '{step.name}': {e}", exc_info=True)
                
                # Handle error step
                if step.on_error:
                    next_step = self._find_step(step.on_error)
                    if next_step is not None:
                        self.current_step = next_step
                    else:
                        raise
                else:
                    raise
        
        logger.info("Workflow execution completed")
        return results
    
    def _execute_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """
        Execute a single workflow step.
        
        Args:
            step: Workflow step
            context: Execution context
        
        Returns:
            Step result
        """
        step_handlers = {
            "load_model": self._load_model_step,
            "train": self._train_step,
            "evaluate": self._evaluate_step,
            "save": self._save_step,
            "inference": self._inference_step,
            "optimize": self._optimize_step,
        }
        
        handler = step_handlers.get(step.step_type)
        if not handler:
            raise ValueError(f"Unknown step type: {step.step_type}")
        
        return handler(step, context)
    
    def _load_model_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Execute load model step."""
        from ..services import ModelService
        
        service = ModelService(registry=self.container._registry)
        service.initialize()
        
        model = service.load_model(step.component, step.config)
        self.container.register("model", model, singleton=True)
        
        return model
    
    def _train_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Execute training step."""
        from ..services import TrainingService
        
        service = TrainingService(registry=self.container._registry)
        service.initialize()
        
        # Get components from container
        model = self.container.get("model")
        train_loader = self.container.get("train_loader")
        optimizer = self.container.get("optimizer")
        scheduler = self.container.get("scheduler")
        scaler = self.container.get("scaler")
        device = self.container.get("device")
        
        service.configure(
            config=step.config,
            model=model,
            train_loader=train_loader,
            val_loader=self.container.get("val_loader"),
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            output_dir=step.config.get("output_dir", "runs/experiment"),
        )
        
        # Train for specified epochs
        epochs = step.config.get("epochs", 1)
        metrics = []
        
        for epoch in range(epochs):
            epoch_metrics = service.train_epoch(
                model, train_loader, optimizer, scheduler, scaler, epoch
            )
            metrics.append(epoch_metrics)
        
        return {"metrics": metrics}
    
    def _evaluate_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Execute evaluation step."""
        from ..services import TrainingService
        
        service = TrainingService(registry=self.container._registry)
        service.initialize()
        
        model = self.container.get("model")
        val_loader = self.container.get("val_loader")
        device = self.container.get("device")
        
        # Configure if needed
        if not service.evaluator:
            from ...training.evaluator import Evaluator
            service.evaluator = Evaluator(device=device)
        
        metrics = service.evaluate(model, val_loader, device)
        return metrics
    
    def _save_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Execute save step."""
        from ..services import ModelService
        
        service = ModelService(registry=self.container._registry)
        service.initialize()
        
        model = self.container.get("model")
        path = step.config.get("path", step.component)
        
        service.save_model(model, path, step.config)
        return {"path": path}
    
    def _inference_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Execute inference step."""
        from ..services import InferenceService
        
        service = InferenceService(registry=self.container._registry)
        service.initialize()
        
        model = self.container.get("model")
        tokenizer = self.container.get("tokenizer")
        
        service.configure(model, tokenizer, step.config)
        
        prompt = step.config.get("prompt", step.component)
        result = service.generate(prompt, step.config.get("generation_config", {}))
        
        return {"result": result}
    
    def _optimize_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Execute optimization step."""
        from ...optimization.performance_optimizer import PerformanceOptimizer
        
        model = self.container.get("model")
        optimizer = PerformanceOptimizer()
        
        optimizations = step.config.get("optimizations", ["torch_compile"])
        optimized = optimizer.optimize_model(model, optimizations)
        
        self.container.register("model", optimized, singleton=True)
        return {"optimizations": optimizations}
    
    def _find_step(self, step_name: str) -> Optional[int]:
        """Find step index by name."""
        for i, step in enumerate(self.steps):
            if step.name == step_name:
                return i
        return None



