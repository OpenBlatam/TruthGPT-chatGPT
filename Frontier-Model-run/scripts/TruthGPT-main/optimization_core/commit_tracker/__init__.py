"""
Advanced Commit Tracking System for TruthGPT Optimization Core
Deep Learning Enhanced Commit Tracking with Performance Analytics

This module provides comprehensive commit tracking capabilities with:
- Deep learning enhanced performance prediction
- Advanced version management with model checkpointing
- Optimization registry with benchmarking
- Interactive web interface with Gradio
- Experiment tracking with wandb/tensorboard
- GPU utilization and mixed precision support
"""

from .commit_tracker import (
    CommitTracker,
    OptimizationCommit,
    CommitStatus,
    CommitType,
    CommitDataset,
    CommitPerformancePredictor,
    create_commit_tracker,
    track_optimization_commit,
    get_commit_history,
    get_commit_statistics
)

from .version_manager import (
    VersionManager,
    VersionInfo,
    VersionType,
    VersionStatus,
    ModelCheckpoint,
    create_version_manager,
    create_version,
    get_version_info,
    get_version_history
)

from .optimization_registry import (
    OptimizationRegistry,
    OptimizationEntry,
    RegistryStatus,
    OptimizationCategory,
    PerformanceProfiler,
    create_optimization_registry,
    register_optimization,
    get_optimization_entry,
    get_registry_statistics
)

# Web Interface
from .gradio_interface import (
    CommitTrackingInterface,
    launch_interface
)

# Streamlit Interface
from .streamlit_interface import main as streamlit_main

# Advanced Libraries
from .advanced_libraries import (
    AdvancedLibraryIntegration,
    AdvancedCommitTracker,
    AdvancedModelOptimizer,
    AdvancedDataProcessor,
    AdvancedVisualization,
    AdvancedAPIServer,
    create_advanced_library_integration,
    create_advanced_commit_tracker,
    create_advanced_model_optimizer,
    create_advanced_data_processor,
    create_advanced_visualization,
    create_advanced_api_server
)

__all__ = [
    # Commit Tracker
    'CommitTracker',
    'OptimizationCommit',
    'CommitStatus',
    'CommitType',
    'CommitDataset',
    'CommitPerformancePredictor',
    'create_commit_tracker',
    'track_optimization_commit',
    'get_commit_history',
    'get_commit_statistics',
    
    # Version Manager
    'VersionManager',
    'VersionInfo',
    'VersionType',
    'VersionStatus',
    'ModelCheckpoint',
    'create_version_manager',
    'create_version',
    'get_version_info',
    'get_version_history',
    
    # Optimization Registry
    'OptimizationRegistry',
    'OptimizationEntry',
    'RegistryStatus',
    'OptimizationCategory',
    'PerformanceProfiler',
    'create_optimization_registry',
    'register_optimization',
    'get_optimization_entry',
    'get_registry_statistics',
    
    # Web Interface
    'CommitTrackingInterface',
    'launch_interface',
    
    # Streamlit Interface
    'streamlit_main',
    
    # Advanced Libraries
    'AdvancedLibraryIntegration',
    'AdvancedCommitTracker',
    'AdvancedModelOptimizer',
    'AdvancedDataProcessor',
    'AdvancedVisualization',
    'AdvancedAPIServer',
    'create_advanced_library_integration',
    'create_advanced_commit_tracker',
    'create_advanced_model_optimizer',
    'create_advanced_data_processor',
    'create_advanced_visualization',
    'create_advanced_api_server'
]

__version__ = "2.0.0"
__author__ = "TruthGPT Optimization Core Team"
__description__ = "Advanced commit tracking with deep learning integration"
