"""
Best Libraries - Curated collection of the best optimization libraries
Implements the most advanced and cutting-edge optimization libraries available
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import time
import logging
import numpy as np
from collections import defaultdict, deque
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial, lru_cache
import gc
import psutil
from contextlib import contextmanager
import warnings
import math
import random
from enum import Enum
import hashlib
import json
import pickle
from pathlib import Path

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class LibraryCategory(Enum):
    """Categories of optimization libraries."""
    DEEP_LEARNING = "deep_learning"
    OPTIMIZATION = "optimization"
    SCIENTIFIC = "scientific"
    GPU_ACCELERATION = "gpu_acceleration"
    DISTRIBUTED = "distributed"
    VISUALIZATION = "visualization"
    MONITORING = "monitoring"
    QUANTUM = "quantum"
    AI_ML = "ai_ml"
    PRODUCTION = "production"

@dataclass
class LibraryInfo:
    """Information about an optimization library."""
    name: str
    category: LibraryCategory
    version: str
    description: str
    features: List[str]
    performance_rating: float
    ease_of_use: float
    documentation_quality: float
    community_support: float
    installation_difficulty: float
    dependencies: List[str] = field(default_factory=list)
    use_cases: List[str] = field(default_factory=list)
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)

class BestLibraries:
    """Curated collection of the best optimization libraries."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.libraries = self._initialize_libraries()
        self.recommendations = defaultdict(list)
        self.performance_history = deque(maxlen=1000)
    
    def _initialize_libraries(self) -> Dict[str, LibraryInfo]:
        """Initialize the best optimization libraries."""
        libraries = {}
        
        # Deep Learning Libraries
        libraries['pytorch'] = LibraryInfo(
            name="PyTorch",
            category=LibraryCategory.DEEP_LEARNING,
            version="2.1.0",
            description="Dynamic neural networks with strong GPU acceleration",
            features=[
                "Dynamic computation graphs",
                "Strong GPU acceleration",
                "Automatic differentiation",
                "Rich ecosystem",
                "Research-friendly",
                "Production-ready"
            ],
            performance_rating=9.5,
            ease_of_use=9.0,
            documentation_quality=9.5,
            community_support=9.8,
            installation_difficulty=3.0,
            dependencies=["torch", "torchvision", "torchaudio"],
            use_cases=["Research", "Production", "Prototyping", "Computer Vision", "NLP"],
            pros=["Excellent documentation", "Strong community", "Flexible", "Fast"],
            cons=["Memory usage", "Learning curve"],
            alternatives=["TensorFlow", "JAX", "Flax"]
        )
        
        libraries['tensorflow'] = LibraryInfo(
            name="TensorFlow",
            category=LibraryCategory.DEEP_LEARNING,
            version="2.15.0",
            description="End-to-end machine learning platform",
            features=[
                "Static computation graphs",
                "TensorFlow Lite",
                "TensorFlow.js",
                "TensorBoard",
                "Keras integration",
                "Production deployment"
            ],
            performance_rating=9.0,
            ease_of_use=8.5,
            documentation_quality=9.0,
            community_support=9.5,
            installation_difficulty=4.0,
            dependencies=["tensorflow", "tensorboard", "tensorflow-probability"],
            use_cases=["Production", "Mobile", "Web", "Large-scale"],
            pros=["Production-ready", "Mobile support", "Comprehensive"],
            cons=["Complexity", "Learning curve"],
            alternatives=["PyTorch", "JAX", "Flax"]
        )
        
        libraries['jax'] = LibraryInfo(
            name="JAX",
            category=LibraryCategory.DEEP_LEARNING,
            version="0.4.20",
            description="NumPy-compatible library for high-performance ML research",
            features=[
                "NumPy-compatible",
                "JIT compilation",
                "Automatic differentiation",
                "Vectorization",
                "Parallelization",
                "Functional programming"
            ],
            performance_rating=9.8,
            ease_of_use=7.5,
            documentation_quality=8.5,
            community_support=8.0,
            installation_difficulty=4.5,
            dependencies=["jax", "jaxlib", "flax", "optax"],
            use_cases=["Research", "High-performance computing", "Scientific computing"],
            pros=["Extremely fast", "NumPy-compatible", "JIT compilation"],
            cons=["Learning curve", "Limited ecosystem"],
            alternatives=["PyTorch", "TensorFlow", "Flax"]
        )
        
        # Optimization Libraries
        libraries['optuna'] = LibraryInfo(
            name="Optuna",
            category=LibraryCategory.OPTIMIZATION,
            version="3.4.0",
            description="Hyperparameter optimization framework",
            features=[
                "TPE sampler",
                "CMA-ES sampler",
                "Pruning",
                "Parallel optimization",
                "Visualization",
                "Integration with ML frameworks"
            ],
            performance_rating=9.5,
            ease_of_use=9.0,
            documentation_quality=9.5,
            community_support=8.5,
            installation_difficulty=2.0,
            dependencies=["optuna", "plotly", "matplotlib"],
            use_cases=["Hyperparameter tuning", "AutoML", "Research"],
            pros=["Easy to use", "Excellent documentation", "Flexible"],
            cons=["Limited to hyperparameter tuning"],
            alternatives=["Hyperopt", "Scikit-optimize", "Ray Tune"]
        )
        
        libraries['hyperopt'] = LibraryInfo(
            name="Hyperopt",
            category=LibraryCategory.OPTIMIZATION,
            version="0.2.7",
            description="Distributed hyperparameter optimization",
            features=[
                "TPE algorithm",
                "Random search",
                "Annealing",
                "Distributed optimization",
                "MongoDB integration",
                "Spark integration"
            ],
            performance_rating=8.5,
            ease_of_use=7.5,
            documentation_quality=7.0,
            community_support=7.0,
            installation_difficulty=3.5,
            dependencies=["hyperopt", "pymongo", "scipy"],
            use_cases=["Distributed optimization", "Large-scale tuning"],
            pros=["Distributed", "MongoDB integration"],
            cons=["Documentation", "Learning curve"],
            alternatives=["Optuna", "Scikit-optimize", "Ray Tune"]
        )
        
        libraries['scikit-optimize'] = LibraryInfo(
            name="Scikit-Optimize",
            category=LibraryCategory.OPTIMIZATION,
            version="0.9.0",
            description="Sequential model-based optimization",
            features=[
                "Bayesian optimization",
                "Gaussian processes",
                "Random forest",
                "Gradient boosting",
                "Acquisition functions",
                "Parallel optimization"
            ],
            performance_rating=8.0,
            ease_of_use=8.5,
            documentation_quality=8.0,
            community_support=7.5,
            installation_difficulty=2.5,
            dependencies=["scikit-optimize", "scikit-learn", "scipy"],
            use_cases=["Bayesian optimization", "Expensive functions"],
            pros=["Bayesian optimization", "Easy to use"],
            cons=["Limited algorithms"],
            alternatives=["Optuna", "Hyperopt", "Ray Tune"]
        )
        
        # Scientific Computing Libraries
        libraries['numpy'] = LibraryInfo(
            name="NumPy",
            category=LibraryCategory.SCIENTIFIC,
            version="1.24.0",
            description="Fundamental package for scientific computing",
            features=[
                "N-dimensional arrays",
                "Linear algebra",
                "Fourier transforms",
                "Random number generation",
                "Mathematical functions",
                "C API"
            ],
            performance_rating=9.0,
            ease_of_use=9.5,
            documentation_quality=9.0,
            community_support=9.8,
            installation_difficulty=1.0,
            dependencies=["numpy"],
            use_cases=["Scientific computing", "Data analysis", "Machine learning"],
            pros=["Fundamental", "Fast", "Well-documented"],
            cons=["Limited to arrays"],
            alternatives=["CuPy", "JAX", "TensorFlow"]
        )
        
        libraries['scipy'] = LibraryInfo(
            name="SciPy",
            category=LibraryCategory.SCIENTIFIC,
            version="1.11.0",
            description="Scientific computing library",
            features=[
                "Optimization",
                "Linear algebra",
                "Statistics",
                "Signal processing",
                "Image processing",
                "Sparse matrices"
            ],
            performance_rating=9.0,
            ease_of_use=8.5,
            documentation_quality=9.0,
            community_support=9.5,
            installation_difficulty=2.0,
            dependencies=["scipy", "numpy"],
            use_cases=["Scientific computing", "Optimization", "Statistics"],
            pros=["Comprehensive", "Well-tested", "Fast"],
            cons=["Large package"],
            alternatives=["CuPy", "JAX", "TensorFlow"]
        )
        
        libraries['cupy'] = LibraryInfo(
            name="CuPy",
            category=LibraryCategory.GPU_ACCELERATION,
            version="12.0.0",
            description="NumPy-compatible GPU array library",
            features=[
                "NumPy-compatible",
                "GPU acceleration",
                "CUDA support",
                "ROCm support",
                "Memory management",
                "Multi-GPU support"
            ],
            performance_rating=9.5,
            ease_of_use=8.0,
            documentation_quality=8.5,
            community_support=8.0,
            installation_difficulty=4.0,
            dependencies=["cupy-cuda12x", "numpy"],
            use_cases=["GPU computing", "Large-scale data", "Scientific computing"],
            pros=["GPU acceleration", "NumPy-compatible"],
            cons=["CUDA dependency", "Memory usage"],
            alternatives=["JAX", "TensorFlow", "PyTorch"]
        )
        
        # Distributed Computing Libraries
        libraries['ray'] = LibraryInfo(
            name="Ray",
            category=LibraryCategory.DISTRIBUTED,
            version="2.8.0",
            description="Distributed computing framework",
            features=[
                "Distributed training",
                "Hyperparameter tuning",
                "Reinforcement learning",
                "Model serving",
                "Workload orchestration",
                "Multi-node support"
            ],
            performance_rating=9.0,
            ease_of_use=7.5,
            documentation_quality=8.5,
            community_support=8.5,
            installation_difficulty=3.5,
            dependencies=["ray[tune]", "ray[rllib]", "ray[serve]"],
            use_cases=["Distributed training", "Hyperparameter tuning", "RL"],
            pros=["Comprehensive", "Scalable", "Production-ready"],
            cons=["Complexity", "Learning curve"],
            alternatives=["Dask", "Horovod", "PyTorch DDP"]
        )
        
        libraries['dask'] = LibraryInfo(
            name="Dask",
            category=LibraryCategory.DISTRIBUTED,
            version="2023.12.0",
            description="Parallel computing library",
            features=[
                "Parallel arrays",
                "DataFrames",
                "Bags",
                "Delayed computation",
                "Distributed computing",
                "Task scheduling"
            ],
            performance_rating=8.5,
            ease_of_use=8.0,
            documentation_quality=9.0,
            community_support=8.0,
            installation_difficulty=3.0,
            dependencies=["dask", "dask[complete]"],
            use_cases=["Parallel computing", "Big data", "Scientific computing"],
            pros=["Easy to use", "NumPy-compatible", "Flexible"],
            cons=["Limited ML support"],
            alternatives=["Ray", "Horovod", "PyTorch DDP"]
        )
        
        # Visualization Libraries
        libraries['matplotlib'] = LibraryInfo(
            name="Matplotlib",
            category=LibraryCategory.VISUALIZATION,
            version="3.8.0",
            description="Comprehensive plotting library",
            features=[
                "2D plotting",
                "3D plotting",
                "Animation",
                "Interactive plots",
                "Publication quality",
                "Customizable"
            ],
            performance_rating=8.5,
            ease_of_use=8.0,
            documentation_quality=9.0,
            community_support=9.5,
            installation_difficulty=2.0,
            dependencies=["matplotlib", "numpy"],
            use_cases=["Data visualization", "Scientific plotting", "Reports"],
            pros=["Comprehensive", "Publication quality", "Flexible"],
            cons=["Performance", "Learning curve"],
            alternatives=["Plotly", "Seaborn", "Bokeh"]
        )
        
        libraries['plotly'] = LibraryInfo(
            name="Plotly",
            category=LibraryCategory.VISUALIZATION,
            version="5.17.0",
            description="Interactive plotting library",
            features=[
                "Interactive plots",
                "3D visualization",
                "Dash integration",
                "Web-based",
                "Real-time updates",
                "Collaboration"
            ],
            performance_rating=9.0,
            ease_of_use=9.0,
            documentation_quality=9.5,
            community_support=8.5,
            installation_difficulty=2.5,
            dependencies=["plotly", "dash"],
            use_cases=["Interactive visualization", "Web apps", "Dashboards"],
            pros=["Interactive", "Web-based", "Easy to use"],
            cons=["Performance", "Dependency"],
            alternatives=["Matplotlib", "Seaborn", "Bokeh"]
        )
        
        # Monitoring Libraries
        libraries['wandb'] = LibraryInfo(
            name="Weights & Biases",
            category=LibraryCategory.MONITORING,
            version="0.16.0",
            description="Experiment tracking and monitoring",
            features=[
                "Experiment tracking",
                "Model versioning",
                "Hyperparameter tuning",
                "Visualization",
                "Collaboration",
                "Production monitoring"
            ],
            performance_rating=9.5,
            ease_of_use=9.5,
            documentation_quality=9.5,
            community_support=9.0,
            installation_difficulty=2.0,
            dependencies=["wandb"],
            use_cases=["Experiment tracking", "Model monitoring", "Collaboration"],
            pros=["Excellent UI", "Easy to use", "Comprehensive"],
            cons=["Cloud dependency", "Cost"],
            alternatives=["TensorBoard", "MLflow", "Neptune"]
        )
        
        libraries['tensorboard'] = LibraryInfo(
            name="TensorBoard",
            category=LibraryCategory.MONITORING,
            version="2.15.0",
            description="TensorFlow visualization toolkit",
            features=[
                "Scalar visualization",
                "Histogram visualization",
                "Image visualization",
                "Graph visualization",
                "Profiling",
                "Embedding visualization"
            ],
            performance_rating=8.5,
            ease_of_use=8.0,
            documentation_quality=8.5,
            community_support=8.5,
            installation_difficulty=2.5,
            dependencies=["tensorboard", "tensorflow"],
            use_cases=["TensorFlow monitoring", "Model visualization", "Profiling"],
            pros=["Integrated", "Comprehensive", "Free"],
            cons=["TensorFlow-centric", "Limited features"],
            alternatives=["Weights & Biases", "MLflow", "Neptune"]
        )
        
        # Quantum Computing Libraries
        libraries['qiskit'] = LibraryInfo(
            name="Qiskit",
            category=LibraryCategory.QUANTUM,
            version="0.45.0",
            description="Quantum computing framework",
            features=[
                "Quantum circuits",
                "Quantum algorithms",
                "Quantum machine learning",
                "Quantum optimization",
                "Quantum simulation",
                "Hardware integration"
            ],
            performance_rating=9.0,
            ease_of_use=7.5,
            documentation_quality=9.0,
            community_support=8.5,
            installation_difficulty=4.0,
            dependencies=["qiskit", "qiskit-aer", "qiskit-optimization"],
            use_cases=["Quantum computing", "Quantum ML", "Research"],
            pros=["Comprehensive", "Well-documented", "Active development"],
            cons=["Complexity", "Limited hardware"],
            alternatives=["Cirq", "PennyLane", "Q#"]
        )
        
        libraries['cirq'] = LibraryInfo(
            name="Cirq",
            category=LibraryCategory.QUANTUM,
            version="1.2.0",
            description="Quantum computing framework by Google",
            features=[
                "Quantum circuits",
                "Quantum algorithms",
                "Quantum simulation",
                "Hardware integration",
                "Research tools",
                "Google Quantum AI"
            ],
            performance_rating=8.5,
            ease_of_use=7.0,
            documentation_quality=8.5,
            community_support=8.0,
            installation_difficulty=4.0,
            dependencies=["cirq", "cirq-google"],
            use_cases=["Quantum research", "Google Quantum AI", "Algorithms"],
            pros=["Google support", "Research-focused", "Hardware integration"],
            cons=["Limited ecosystem", "Complexity"],
            alternatives=["Qiskit", "PennyLane", "Q#"]
        )
        
        # AI/ML Libraries
        libraries['scikit-learn'] = LibraryInfo(
            name="Scikit-learn",
            category=LibraryCategory.AI_ML,
            version="1.3.0",
            description="Machine learning library",
            features=[
                "Classification",
                "Regression",
                "Clustering",
                "Dimensionality reduction",
                "Model selection",
                "Preprocessing"
            ],
            performance_rating=8.5,
            ease_of_use=9.5,
            documentation_quality=9.5,
            community_support=9.8,
            installation_difficulty=1.5,
            dependencies=["scikit-learn", "numpy", "scipy"],
            use_cases=["Machine learning", "Data science", "Research"],
            pros=["Easy to use", "Comprehensive", "Well-documented"],
            cons=["Limited deep learning", "Performance"],
            alternatives=["TensorFlow", "PyTorch", "XGBoost"]
        )
        
        libraries['xgboost'] = LibraryInfo(
            name="XGBoost",
            category=LibraryCategory.AI_ML,
            version="2.0.0",
            description="Gradient boosting framework",
            features=[
                "Gradient boosting",
                "Tree-based models",
                "GPU acceleration",
                "Distributed training",
                "Feature importance",
                "Model interpretation"
            ],
            performance_rating=9.5,
            ease_of_use=8.5,
            documentation_quality=8.5,
            community_support=9.0,
            installation_difficulty=2.5,
            dependencies=["xgboost"],
            use_cases=["Tabular data", "Competitions", "Production"],
            pros=["High performance", "GPU support", "Feature importance"],
            cons=["Limited to tree models", "Memory usage"],
            alternatives=["LightGBM", "CatBoost", "Scikit-learn"]
        )
        
        # Production Libraries
        libraries['fastapi'] = LibraryInfo(
            name="FastAPI",
            category=LibraryCategory.PRODUCTION,
            version="0.104.0",
            description="Modern web framework for APIs",
            features=[
                "Fast performance",
                "Automatic documentation",
                "Type hints",
                "Async support",
                "Validation",
                "OpenAPI"
            ],
            performance_rating=9.5,
            ease_of_use=9.0,
            documentation_quality=9.5,
            community_support=9.0,
            installation_difficulty=2.0,
            dependencies=["fastapi", "uvicorn"],
            use_cases=["API development", "ML serving", "Microservices"],
            pros=["Fast", "Easy to use", "Automatic docs"],
            cons=["Limited ecosystem", "New framework"],
            alternatives=["Flask", "Django", "FastAPI"]
        )
        
        libraries['mlflow'] = LibraryInfo(
            name="MLflow",
            category=LibraryCategory.PRODUCTION,
            version="2.8.0",
            description="ML lifecycle management",
            features=[
                "Experiment tracking",
                "Model registry",
                "Model deployment",
                "Model versioning",
                "Collaboration",
                "Production monitoring"
            ],
            performance_rating=9.0,
            ease_of_use=8.5,
            documentation_quality=9.0,
            community_support=8.5,
            installation_difficulty=3.0,
            dependencies=["mlflow"],
            use_cases=["ML lifecycle", "Model management", "Production"],
            pros=["Comprehensive", "Open source", "Flexible"],
            cons=["Complexity", "Setup"],
            alternatives=["Weights & Biases", "Neptune", "DVC"]
        )
        
        return libraries
    
    def get_library(self, name: str) -> Optional[LibraryInfo]:
        """Get information about a specific library."""
        return self.libraries.get(name)
    
    def get_libraries_by_category(self, category: LibraryCategory) -> List[LibraryInfo]:
        """Get libraries by category."""
        return [lib for lib in self.libraries.values() if lib.category == category]
    
    def get_top_libraries(self, category: Optional[LibraryCategory] = None, 
                         limit: int = 10) -> List[LibraryInfo]:
        """Get top libraries by performance rating."""
        libraries = self.libraries.values()
        
        if category:
            libraries = [lib for lib in libraries if lib.category == category]
        
        # Sort by performance rating
        sorted_libraries = sorted(libraries, key=lambda x: x.performance_rating, reverse=True)
        
        return sorted_libraries[:limit]
    
    def get_recommendations(self, use_case: str, 
                          requirements: List[str] = None) -> List[LibraryInfo]:
        """Get library recommendations based on use case and requirements."""
        recommendations = []
        
        # Use case mapping
        use_case_mapping = {
            'deep_learning': ['pytorch', 'tensorflow', 'jax'],
            'optimization': ['optuna', 'hyperopt', 'scikit-optimize'],
            'scientific': ['numpy', 'scipy', 'cupy'],
            'distributed': ['ray', 'dask'],
            'visualization': ['matplotlib', 'plotly'],
            'monitoring': ['wandb', 'tensorboard'],
            'quantum': ['qiskit', 'cirq'],
            'ml': ['scikit-learn', 'xgboost'],
            'production': ['fastapi', 'mlflow']
        }
        
        if use_case in use_case_mapping:
            for lib_name in use_case_mapping[use_case]:
                if lib_name in self.libraries:
                    recommendations.append(self.libraries[lib_name])
        
        # Filter by requirements if provided
        if requirements:
            filtered_recommendations = []
            for lib in recommendations:
                if any(req.lower() in lib.description.lower() or 
                      any(req.lower() in feature.lower() for feature in lib.features)
                      for req in requirements):
                    filtered_recommendations.append(lib)
            recommendations = filtered_recommendations
        
        # Sort by performance rating
        recommendations.sort(key=lambda x: x.performance_rating, reverse=True)
        
        return recommendations
    
    def compare_libraries(self, library_names: List[str]) -> Dict[str, Any]:
        """Compare multiple libraries."""
        if not library_names:
            return {}
        
        libraries = [self.libraries[name] for name in library_names if name in self.libraries]
        
        if not libraries:
            return {}
        
        comparison = {
            'libraries': [lib.name for lib in libraries],
            'performance_ratings': [lib.performance_rating for lib in libraries],
            'ease_of_use': [lib.ease_of_use for lib in libraries],
            'documentation_quality': [lib.documentation_quality for lib in libraries],
            'community_support': [lib.community_support for lib in libraries],
            'installation_difficulty': [lib.installation_difficulty for lib in libraries],
            'features': [lib.features for lib in libraries],
            'pros': [lib.pros for lib in libraries],
            'cons': [lib.cons for lib in libraries]
        }
        
        return comparison
    
    def get_installation_guide(self, library_name: str) -> Dict[str, Any]:
        """Get installation guide for a library."""
        library = self.get_library(library_name)
        if not library:
            return {}
        
        guide = {
            'name': library.name,
            'version': library.version,
            'dependencies': library.dependencies,
            'installation_commands': self._generate_installation_commands(library),
            'verification_commands': self._generate_verification_commands(library),
            'troubleshooting': self._generate_troubleshooting_guide(library)
        }
        
        return guide
    
    def _generate_installation_commands(self, library: LibraryInfo) -> List[str]:
        """Generate installation commands for a library."""
        commands = []
        
        # Base installation
        if library.name.lower() in ['pytorch', 'tensorflow']:
            commands.append(f"# Install {library.name}")
            commands.append(f"pip install {library.name}")
        else:
            commands.append(f"pip install {' '.join(library.dependencies)}")
        
        # Additional setup for specific libraries
        if library.name == 'PyTorch':
            commands.append("# For CUDA support:")
            commands.append("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        elif library.name == 'CuPy':
            commands.append("# For CUDA 12.x:")
            commands.append("pip install cupy-cuda12x")
        elif library.name == 'Ray':
            commands.append("# For full Ray installation:")
            commands.append("pip install ray[tune,rllib,serve]")
        
        return commands
    
    def _generate_verification_commands(self, library: LibraryInfo) -> List[str]:
        """Generate verification commands for a library."""
        commands = []
        
        commands.append(f"# Verify {library.name} installation")
        commands.append(f"python -c \"import {library.name.lower().replace('-', '_')}; print('Installation successful')\"")
        
        # Specific verification for certain libraries
        if library.name == 'PyTorch':
            commands.append("python -c \"import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')\"")
        elif library.name == 'TensorFlow':
            commands.append("python -c \"import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}'); print(f'GPU available: {tf.config.list_physical_devices(\"GPU\")}')\"")
        elif library.name == 'CuPy':
            commands.append("python -c \"import cupy; print(f'CuPy version: {cupy.__version__}'); print(f'CUDA version: {cupy.cuda.runtime.runtimeGetVersion()}')\"")
        
        return commands
    
    def _generate_troubleshooting_guide(self, library: LibraryInfo) -> List[str]:
        """Generate troubleshooting guide for a library."""
        guide = []
        
        guide.append(f"# Troubleshooting {library.name}")
        guide.append("")
        
        # Common issues
        if library.name == 'PyTorch':
            guide.append("## Common Issues:")
            guide.append("- CUDA not available: Check CUDA installation and PyTorch CUDA version")
            guide.append("- Memory issues: Reduce batch size or use gradient checkpointing")
            guide.append("- Performance issues: Enable cuDNN benchmark mode")
        elif library.name == 'TensorFlow':
            guide.append("## Common Issues:")
            guide.append("- GPU not detected: Check CUDA/cuDNN installation")
            guide.append("- Memory issues: Configure GPU memory growth")
            guide.append("- Version conflicts: Use virtual environments")
        elif library.name == 'CuPy':
            guide.append("## Common Issues:")
            guide.append("- CUDA version mismatch: Install correct CuPy version")
            guide.append("- Memory issues: Check GPU memory availability")
            guide.append("- Installation fails: Install CUDA toolkit first")
        
        guide.append("")
        guide.append("## Getting Help:")
        guide.append(f"- Documentation: Check {library.name} official docs")
        guide.append("- Community: GitHub issues, Stack Overflow")
        guide.append("- Support: Official forums and communities")
        
        return guide
    
    def get_library_statistics(self) -> Dict[str, Any]:
        """Get statistics about all libraries."""
        libraries = list(self.libraries.values())
        
        return {
            'total_libraries': len(libraries),
            'categories': len(LibraryCategory),
            'avg_performance_rating': np.mean([lib.performance_rating for lib in libraries]),
            'avg_ease_of_use': np.mean([lib.ease_of_use for lib in libraries]),
            'avg_documentation_quality': np.mean([lib.documentation_quality for lib in libraries]),
            'avg_community_support': np.mean([lib.community_support for lib in libraries]),
            'avg_installation_difficulty': np.mean([lib.installation_difficulty for lib in libraries]),
            'top_performers': [lib.name for lib in sorted(libraries, key=lambda x: x.performance_rating, reverse=True)[:5]],
            'easiest_to_use': [lib.name for lib in sorted(libraries, key=lambda x: x.ease_of_use, reverse=True)[:5]],
            'best_documented': [lib.name for lib in sorted(libraries, key=lambda x: x.documentation_quality, reverse=True)[:5]]
        }

# Factory functions
def create_best_libraries(config: Optional[Dict[str, Any]] = None) -> BestLibraries:
    """Create best libraries instance."""
    return BestLibraries(config)

@contextmanager
def best_libraries_context(config: Optional[Dict[str, Any]] = None):
    """Context manager for best libraries."""
    libraries = create_best_libraries(config)
    try:
        yield libraries
    finally:
        # Cleanup if needed
        pass

