"""
Compilation Pipeline for TruthGPT Compiler
Pipeline management for compilation stages
"""

import logging
import time
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PipelineStage:
    """Represents a stage in the compilation pipeline"""
    name: str
    description: str
    enabled: bool = True
    priority: int = 0
    timeout: Optional[float] = None
    dependencies: List[str] = None
    parameters: Dict[str, Any] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.parameters is None:
            self.parameters = {}

@dataclass
class PipelineResult:
    """Result of pipeline execution"""
    success: bool
    stage_results: Dict[str, Any] = None
    total_time: float = 0.0
    errors: List[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.stage_results is None:
            self.stage_results = {}
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []

class CompilationPipeline:
    """Manages compilation pipeline stages"""
    
    def __init__(self, name: str):
        self.name = name
        self.stages = {}
        self.stage_dependencies = {}
        self.execution_order = []
        
    def add_stage(self, stage: PipelineStage, stage_func: Callable):
        """Add a stage to the pipeline"""
        self.stages[stage.name] = {
            "config": stage,
            "function": stage_func
        }
        self.stage_dependencies[stage.name] = stage.dependencies
        self._update_execution_order()
        
    def _update_execution_order(self):
        """Update stage execution order based on dependencies"""
        # Topological sort based on dependencies
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(stage_name):
            if stage_name in temp_visited:
                raise ValueError(f"Circular dependency detected: {stage_name}")
            if stage_name in visited:
                return
            
            temp_visited.add(stage_name)
            for dep in self.stage_dependencies.get(stage_name, []):
                if dep in self.stages:
                    visit(dep)
            temp_visited.remove(stage_name)
            visited.add(stage_name)
            order.append(stage_name)
        
        for stage_name in self.stages:
            if stage_name not in visited:
                visit(stage_name)
        
        self.execution_order = order
    
    def execute(self, input_data: Any, **kwargs) -> PipelineResult:
        """Execute the compilation pipeline"""
        start_time = time.time()
        stage_results = {}
        errors = []
        warnings = []
        
        try:
            current_data = input_data
            
            for stage_name in self.execution_order:
                stage_info = self.stages[stage_name]
                stage_config = stage_info["config"]
                stage_func = stage_info["function"]
                
                if not stage_config.enabled:
                    logger.info(f"Skipping disabled stage: {stage_name}")
                    continue
                
                logger.info(f"Executing stage: {stage_name}")
                stage_start_time = time.time()
                
                try:
                    # Execute stage
                    result = stage_func(current_data, **kwargs)
                    stage_results[stage_name] = result
                    current_data = result
                    
                    stage_time = time.time() - stage_start_time
                    logger.info(f"Stage {stage_name} completed in {stage_time:.3f}s")
                    
                except Exception as e:
                    error_msg = f"Stage {stage_name} failed: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    
                    # Continue with next stage if possible
                    continue
            
            total_time = time.time() - start_time
            
            return PipelineResult(
                success=len(errors) == 0,
                stage_results=stage_results,
                total_time=total_time,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            total_time = time.time() - start_time
            error_msg = f"Pipeline execution failed: {str(e)}"
            logger.error(error_msg)
            
            return PipelineResult(
                success=False,
                stage_results=stage_results,
                total_time=total_time,
                errors=[error_msg],
                warnings=warnings
            )
    
    def get_stage_info(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a stage"""
        if stage_name in self.stages:
            stage_info = self.stages[stage_name]
            return {
                "name": stage_name,
                "description": stage_info["config"].description,
                "enabled": stage_info["config"].enabled,
                "priority": stage_info["config"].priority,
                "dependencies": stage_info["config"].dependencies
            }
        return None
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline information"""
        return {
            "name": self.name,
            "total_stages": len(self.stages),
            "enabled_stages": sum(1 for stage_info in self.stages.values() 
                                if stage_info["config"].enabled),
            "execution_order": self.execution_order
        }

def create_compilation_pipeline(name: str) -> CompilationPipeline:
    """Create a compilation pipeline"""
    return CompilationPipeline(name)

def pipeline_context(pipeline: CompilationPipeline):
    """Create a pipeline execution context"""
    class PipelineContext:
        def __init__(self, pipe: CompilationPipeline):
            self.pipeline = pipe
            
        def __enter__(self):
            logger.info(f"Starting pipeline: {self.pipeline.name}")
            return self.pipeline
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            logger.info(f"Pipeline {self.pipeline.name} completed")
    
    return PipelineContext(pipeline)






