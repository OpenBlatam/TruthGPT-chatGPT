"""
Configuration for VR/AR Optimization Engine.
"""
from dataclasses import dataclass, field
from typing import List
from optimization_core.modules.vr_optimization.enums import VROptimizationLevel, ImmersiveTechnology

@dataclass
class VROptimizationConfig:
    """VR/AR optimization configuration."""
    level: VROptimizationLevel = VROptimizationLevel.VR_ADVANCED
    immersive_technologies: List[ImmersiveTechnology] = field(default_factory=lambda: [ImmersiveTechnology.VIRTUAL_REALITY])
    enable_immersive_rendering: bool = True
    enable_spatial_computing: bool = True
    enable_haptic_feedback: bool = True
    enable_eye_tracking: bool = True
    enable_hand_tracking: bool = True
    enable_voice_recognition: bool = True
    enable_gesture_recognition: bool = True
    enable_emotion_detection: bool = True
    enable_brain_interface: bool = True
    enable_neural_interface: bool = True
    enable_consciousness_simulation: bool = True
    enable_digital_immortality: bool = True
    enable_metaverse_integration: bool = True
    enable_synthetic_reality: bool = True
    enable_transcendental_reality: bool = True
    enable_divine_reality: bool = True
    enable_omnipotent_reality: bool = True
    enable_infinite_reality: bool = True
    enable_universal_reality: bool = True
    max_workers: int = 32
    optimization_timeout: float = 600.0
    immersion_depth: int = 1000
    reality_layers: int = 100

