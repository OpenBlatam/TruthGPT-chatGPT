"""
Metaverse Test Framework for TruthGPT Optimization Core
=======================================================

This module implements metaverse testing capabilities including:
- Virtual reality test environments
- Augmented reality test overlays
- Virtual test avatars
- Metaverse test interactions
- Virtual test collaboration
"""

import unittest
import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import random
import math
from datetime import datetime
from collections import defaultdict
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VirtualAvatar:
    """Represents a virtual test avatar"""
    avatar_id: str
    user_id: str
    avatar_name: str
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    appearance: Dict[str, Any]
    capabilities: List[str]
    status: str
    last_activity: datetime

@dataclass
class VirtualTestObject:
    """Represents a test object in virtual space"""
    object_id: str
    test_id: str
    object_type: str
    position: Tuple[float, float, float]
    scale: Tuple[float, float, float]
    properties: Dict[str, Any]
    interactions: List[str]
    state: str
    metadata: Dict[str, Any]

@dataclass
class MetaverseEnvironment:
    """Represents a metaverse test environment"""
    environment_id: str
    name: str
    world_type: str
    dimensions: Tuple[float, float, float]
    physics_settings: Dict[str, Any]
    lighting: Dict[str, Any]
    avatars: List[VirtualAvatar]
    test_objects: List[VirtualTestObject]
    spawn_points: List[Tuple[float, float, float]]
    teleporters: List[Dict[str, Any]]

class VirtualRealityTestEngine:
    """Virtual reality test execution engine"""
    
    def __init__(self):
        self.vr_headsets = {}
        self.vr_environments = {}
        self.active_sessions = {}
        self.haptic_feedback = {}
        self.spatial_tracking = {}
    
    def initialize_vr_session(self, user_id: str, headset_type: str) -> str:
        """Initialize VR test session"""
        logger.info(f"Initializing VR session for user {user_id} with {headset_type}")
        
        session_id = f"vr_session_{user_id}_{int(time.time())}"
        
        # Initialize VR headset
        headset_config = self._get_headset_config(headset_type)
        
        vr_session = {
            "session_id": session_id,
            "user_id": user_id,
            "headset_type": headset_type,
            "headset_config": headset_config,
            "tracking_data": {
                "head_position": (0, 0, 0),
                "head_rotation": (0, 0, 0),
                "hand_positions": [(0, 0, 0), (0, 0, 0)],
                "hand_rotations": [(0, 0, 0), (0, 0, 0)]
            },
            "haptic_intensity": 0.5,
            "render_quality": "high",
            "fps": 90,
            "latency": 20  # ms
        }
        
        self.active_sessions[session_id] = vr_session
        return session_id
    
    def _get_headset_config(self, headset_type: str) -> Dict[str, Any]:
        """Get configuration for VR headset"""
        configs = {
            "oculus_quest": {
                "resolution": (1832, 1920),
                "fov": 100,
                "refresh_rate": 72,
                "tracking": "inside_out",
                "controllers": 2
            },
            "htc_vive": {
                "resolution": (1080, 1200),
                "fov": 110,
                "refresh_rate": 90,
                "tracking": "lighthouse",
                "controllers": 2
            },
            "valve_index": {
                "resolution": (1440, 1600),
                "fov": 130,
                "refresh_rate": 120,
                "tracking": "lighthouse",
                "controllers": 2
            },
            "playstation_vr": {
                "resolution": (960, 1080),
                "fov": 100,
                "refresh_rate": 60,
                "tracking": "camera",
                "controllers": 2
            }
        }
        
        return configs.get(headset_type, configs["oculus_quest"])
    
    def load_vr_test_environment(self, session_id: str, environment: MetaverseEnvironment) -> bool:
        """Load VR test environment"""
        if session_id not in self.active_sessions:
            raise ValueError(f"VR session not found: {session_id}")
        
        logger.info(f"Loading VR environment {environment.name} for session {session_id}")
        
        # Simulate environment loading
        loading_time = random.uniform(2.0, 5.0)
        time.sleep(loading_time)
        
        # Update session with environment
        self.active_sessions[session_id]["environment"] = environment
        
        return True
    
    def execute_vr_test(self, session_id: str, test_object: VirtualTestObject) -> Dict[str, Any]:
        """Execute test in VR environment"""
        if session_id not in self.active_sessions:
            raise ValueError(f"VR session not found: {session_id}")
        
        logger.info(f"Executing VR test {test_object.test_id} for session {session_id}")
        
        session = self.active_sessions[session_id]
        
        # Simulate VR test execution
        execution_time = random.uniform(1.0, 3.0)
        
        # Simulate haptic feedback
        self._simulate_haptic_feedback(session_id, test_object)
        
        # Simulate spatial interactions
        interactions = self._simulate_spatial_interactions(session_id, test_object)
        
        # Generate test result
        result = {
            "test_id": test_object.test_id,
            "execution_time": execution_time,
            "vr_metrics": {
                "frame_rate": session["fps"],
                "latency": session["latency"],
                "tracking_accuracy": random.uniform(0.95, 0.99),
                "haptic_feedback": session["haptic_intensity"]
            },
            "interactions": interactions,
            "success": random.choice([True, False]),
            "timestamp": datetime.now()
        }
        
        return result
    
    def _simulate_haptic_feedback(self, session_id: str, test_object: VirtualTestObject):
        """Simulate haptic feedback during test execution"""
        session = self.active_sessions[session_id]
        
        # Simulate different haptic patterns based on test type
        test_type = test_object.object_type
        
        haptic_patterns = {
            "unit_test": {"duration": 0.1, "intensity": 0.3, "pattern": "pulse"},
            "integration_test": {"duration": 0.3, "intensity": 0.5, "pattern": "vibration"},
            "performance_test": {"duration": 0.5, "intensity": 0.7, "pattern": "wave"},
            "stress_test": {"duration": 1.0, "intensity": 0.9, "pattern": "intense"}
        }
        
        pattern = haptic_patterns.get(test_type, haptic_patterns["unit_test"])
        
        # Simulate haptic feedback
        logger.info(f"Haptic feedback: {pattern['pattern']} for {pattern['duration']}s")
    
    def _simulate_spatial_interactions(self, session_id: str, 
                                     test_object: VirtualTestObject) -> List[Dict[str, Any]]:
        """Simulate spatial interactions with test objects"""
        interactions = []
        
        # Simulate hand tracking interactions
        for hand in ["left", "right"]:
            interaction = {
                "hand": hand,
                "interaction_type": random.choice(["grab", "point", "gesture", "touch"]),
                "object_id": test_object.object_id,
                "position": test_object.position,
                "timestamp": datetime.now()
            }
            interactions.append(interaction)
        
        return interactions

class AugmentedRealityOverlay:
    """Augmented reality test overlay system"""
    
    def __init__(self):
        self.ar_devices = {}
        self.overlay_layers = {}
        self.tracking_systems = {}
        self.anchor_points = {}
    
    def initialize_ar_device(self, device_id: str, device_type: str) -> str:
        """Initialize AR device for testing"""
        logger.info(f"Initializing AR device {device_id} of type {device_type}")
        
        device_config = self._get_ar_device_config(device_type)
        
        ar_device = {
            "device_id": device_id,
            "device_type": device_type,
            "config": device_config,
            "camera_feed": True,
            "tracking_active": True,
            "overlay_layers": [],
            "anchor_points": [],
            "last_update": datetime.now()
        }
        
        self.ar_devices[device_id] = ar_device
        return device_id
    
    def _get_ar_device_config(self, device_type: str) -> Dict[str, Any]:
        """Get configuration for AR device"""
        configs = {
            "hololens": {
                "resolution": (1268, 720),
                "fov": 30,
                "tracking": "inside_out",
                "hand_tracking": True,
                "eye_tracking": True
            },
            "magic_leap": {
                "resolution": (1280, 720),
                "fov": 40,
                "tracking": "inside_out",
                "hand_tracking": True,
                "eye_tracking": False
            },
            "iphone_ar": {
                "resolution": (1920, 1080),
                "fov": 60,
                "tracking": "camera",
                "hand_tracking": False,
                "eye_tracking": False
            },
            "android_ar": {
                "resolution": (1920, 1080),
                "fov": 60,
                "tracking": "camera",
                "hand_tracking": False,
                "eye_tracking": False
            }
        }
        
        return configs.get(device_type, configs["iphone_ar"])
    
    def create_test_overlay(self, device_id: str, test_data: Dict[str, Any]) -> str:
        """Create AR overlay for test data"""
        if device_id not in self.ar_devices:
            raise ValueError(f"AR device not found: {device_id}")
        
        logger.info(f"Creating test overlay for device {device_id}")
        
        overlay_id = f"overlay_{device_id}_{int(time.time())}"
        
        overlay = {
            "overlay_id": overlay_id,
            "device_id": device_id,
            "test_data": test_data,
            "overlay_type": "test_visualization",
            "position": (0, 0, 0),
            "scale": (1, 1, 1),
            "opacity": 0.8,
            "interactive": True,
            "anchor_point": None,
            "created_at": datetime.now()
        }
        
        self.overlay_layers[overlay_id] = overlay
        
        # Add to device
        self.ar_devices[device_id]["overlay_layers"].append(overlay_id)
        
        return overlay_id
    
    def anchor_overlay_to_world(self, overlay_id: str, 
                              anchor_position: Tuple[float, float, float]) -> bool:
        """Anchor overlay to real-world position"""
        if overlay_id not in self.overlay_layers:
            raise ValueError(f"Overlay not found: {overlay_id}")
        
        logger.info(f"Anchoring overlay {overlay_id} to world position {anchor_position}")
        
        overlay = self.overlay_layers[overlay_id]
        overlay["anchor_point"] = anchor_position
        overlay["position"] = anchor_position
        
        return True
    
    def update_overlay_content(self, overlay_id: str, new_data: Dict[str, Any]) -> bool:
        """Update overlay content in real-time"""
        if overlay_id not in self.overlay_layers:
            raise ValueError(f"Overlay not found: {overlay_id}")
        
        logger.info(f"Updating overlay content for {overlay_id}")
        
        overlay = self.overlay_layers[overlay_id]
        overlay["test_data"].update(new_data)
        overlay["last_update"] = datetime.now()
        
        return True
    
    def track_user_interaction(self, device_id: str, 
                             interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Track user interactions with AR overlays"""
        if device_id not in self.ar_devices:
            raise ValueError(f"AR device not found: {device_id}")
        
        logger.info(f"Tracking user interaction for device {device_id}")
        
        # Simulate interaction tracking
        interaction_result = {
            "device_id": device_id,
            "interaction_type": interaction_data.get("type", "unknown"),
            "target_overlay": interaction_data.get("target", None),
            "position": interaction_data.get("position", (0, 0, 0)),
            "timestamp": datetime.now(),
            "success": random.choice([True, False])
        }
        
        return interaction_result

class VirtualTestCollaboration:
    """Virtual test collaboration system"""
    
    def __init__(self):
        self.collaboration_rooms = {}
        self.user_sessions = {}
        self.shared_objects = {}
        self.voice_channels = {}
    
    def create_collaboration_room(self, room_name: str, creator_id: str) -> str:
        """Create virtual collaboration room"""
        logger.info(f"Creating collaboration room '{room_name}' by user {creator_id}")
        
        room_id = f"room_{room_name}_{int(time.time())}"
        
        collaboration_room = {
            "room_id": room_id,
            "room_name": room_name,
            "creator_id": creator_id,
            "participants": [creator_id],
            "max_participants": 10,
            "environment": None,
            "shared_objects": [],
            "voice_channel": None,
            "permissions": {
                "can_modify_objects": True,
                "can_invite_users": True,
                "can_control_environment": True
            },
            "created_at": datetime.now()
        }
        
        self.collaboration_rooms[room_id] = collaboration_room
        return room_id
    
    def join_collaboration_room(self, room_id: str, user_id: str) -> bool:
        """Join virtual collaboration room"""
        if room_id not in self.collaboration_rooms:
            raise ValueError(f"Collaboration room not found: {room_id}")
        
        room = self.collaboration_rooms[room_id]
        
        if len(room["participants"]) >= room["max_participants"]:
            logger.warning(f"Room {room_id} is full")
            return False
        
        if user_id not in room["participants"]:
            room["participants"].append(user_id)
            logger.info(f"User {user_id} joined room {room_id}")
        
        return True
    
    def share_test_object(self, room_id: str, user_id: str, 
                         test_object: VirtualTestObject) -> str:
        """Share test object in collaboration room"""
        if room_id not in self.collaboration_rooms:
            raise ValueError(f"Collaboration room not found: {room_id}")
        
        if user_id not in self.collaboration_rooms[room_id]["participants"]:
            raise ValueError(f"User {user_id} not in room {room_id}")
        
        logger.info(f"Sharing test object {test_object.object_id} in room {room_id}")
        
        shared_object_id = f"shared_{test_object.object_id}_{int(time.time())}"
        
        shared_object = {
            "shared_object_id": shared_object_id,
            "original_object": test_object,
            "owner_id": user_id,
            "room_id": room_id,
            "permissions": {
                "can_view": True,
                "can_modify": False,
                "can_interact": True
            },
            "collaboration_history": [],
            "created_at": datetime.now()
        }
        
        self.shared_objects[shared_object_id] = shared_object
        self.collaboration_rooms[room_id]["shared_objects"].append(shared_object_id)
        
        return shared_object_id
    
    def collaborate_on_test(self, room_id: str, user_id: str, 
                           shared_object_id: str, action: str) -> Dict[str, Any]:
        """Perform collaborative action on shared test object"""
        if room_id not in self.collaboration_rooms:
            raise ValueError(f"Collaboration room not found: {room_id}")
        
        if shared_object_id not in self.shared_objects:
            raise ValueError(f"Shared object not found: {shared_object_id}")
        
        shared_object = self.shared_objects[shared_object_id]
        
        if user_id not in self.collaboration_rooms[room_id]["participants"]:
            raise ValueError(f"User {user_id} not in room {room_id}")
        
        logger.info(f"User {user_id} performing action '{action}' on shared object {shared_object_id}")
        
        # Record collaboration action
        collaboration_action = {
            "user_id": user_id,
            "action": action,
            "timestamp": datetime.now(),
            "object_state": shared_object["original_object"].state,
            "success": random.choice([True, False])
        }
        
        shared_object["collaboration_history"].append(collaboration_action)
        
        return collaboration_action
    
    def setup_voice_channel(self, room_id: str) -> str:
        """Setup voice channel for collaboration room"""
        if room_id not in self.collaboration_rooms:
            raise ValueError(f"Collaboration room not found: {room_id}")
        
        logger.info(f"Setting up voice channel for room {room_id}")
        
        voice_channel_id = f"voice_{room_id}_{int(time.time())}"
        
        voice_channel = {
            "channel_id": voice_channel_id,
            "room_id": room_id,
            "participants": [],
            "audio_quality": "high",
            "spatial_audio": True,
            "noise_cancellation": True,
            "created_at": datetime.now()
        }
        
        self.voice_channels[voice_channel_id] = voice_channel
        self.collaboration_rooms[room_id]["voice_channel"] = voice_channel_id
        
        return voice_channel_id

class MetaverseTestGenerator(unittest.TestCase):
    """Test cases for Metaverse Test Framework"""
    
    def setUp(self):
        self.vr_engine = VirtualRealityTestEngine()
        self.ar_overlay = AugmentedRealityOverlay()
        self.collaboration = VirtualTestCollaboration()
    
    def test_virtual_avatar_creation(self):
        """Test virtual avatar creation"""
        avatar = VirtualAvatar(
            avatar_id="avatar_001",
            user_id="user_001",
            avatar_name="TestAvatar",
            position=(1.0, 2.0, 3.0),
            rotation=(0.0, 0.0, 0.0),
            appearance={"skin_color": "fair", "hair_color": "brown"},
            capabilities=["walk", "jump", "interact"],
            status="active",
            last_activity=datetime.now()
        )
        
        self.assertEqual(avatar.avatar_id, "avatar_001")
        self.assertEqual(avatar.user_id, "user_001")
        self.assertEqual(avatar.position, (1.0, 2.0, 3.0))
        self.assertIn("walk", avatar.capabilities)
    
    def test_virtual_test_object_creation(self):
        """Test virtual test object creation"""
        test_object = VirtualTestObject(
            object_id="test_obj_001",
            test_id="test_001",
            object_type="unit_test",
            position=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
            properties={"complexity": "medium", "duration": 5.0},
            interactions=["click", "drag", "rotate"],
            state="idle",
            metadata={"created_by": "user_001"}
        )
        
        self.assertEqual(test_object.object_id, "test_obj_001")
        self.assertEqual(test_object.object_type, "unit_test")
        self.assertIn("click", test_object.interactions)
    
    def test_metaverse_environment_creation(self):
        """Test metaverse environment creation"""
        environment = MetaverseEnvironment(
            environment_id="env_001",
            name="Test Environment",
            world_type="virtual_lab",
            dimensions=(10.0, 10.0, 10.0),
            physics_settings={"gravity": 9.81, "collision": True},
            lighting={"ambient": 0.3, "directional": 0.7},
            avatars=[],
            test_objects=[],
            spawn_points=[(0, 0, 0), (5, 0, 5)],
            teleporters=[]
        )
        
        self.assertEqual(environment.environment_id, "env_001")
        self.assertEqual(environment.world_type, "virtual_lab")
        self.assertEqual(len(environment.spawn_points), 2)
    
    def test_vr_session_initialization(self):
        """Test VR session initialization"""
        session_id = self.vr_engine.initialize_vr_session("user_001", "oculus_quest")
        
        self.assertIsNotNone(session_id)
        self.assertIn(session_id, self.vr_engine.active_sessions)
        
        session = self.vr_engine.active_sessions[session_id]
        self.assertEqual(session["user_id"], "user_001")
        self.assertEqual(session["headset_type"], "oculus_quest")
    
    def test_vr_headset_configurations(self):
        """Test VR headset configurations"""
        headset_types = ["oculus_quest", "htc_vive", "valve_index", "playstation_vr"]
        
        for headset_type in headset_types:
            config = self.vr_engine._get_headset_config(headset_type)
            
            self.assertIsInstance(config, dict)
            self.assertIn("resolution", config)
            self.assertIn("fov", config)
            self.assertIn("refresh_rate", config)
            self.assertIn("tracking", config)
    
    def test_vr_environment_loading(self):
        """Test VR environment loading"""
        session_id = self.vr_engine.initialize_vr_session("user_001", "oculus_quest")
        
        environment = MetaverseEnvironment(
            environment_id="test_env",
            name="Test Environment",
            world_type="virtual_lab",
            dimensions=(10, 10, 10),
            physics_settings={},
            lighting={},
            avatars=[],
            test_objects=[],
            spawn_points=[],
            teleporters=[]
        )
        
        success = self.vr_engine.load_vr_test_environment(session_id, environment)
        
        self.assertTrue(success)
        self.assertIn("environment", self.vr_engine.active_sessions[session_id])
    
    def test_vr_test_execution(self):
        """Test VR test execution"""
        session_id = self.vr_engine.initialize_vr_session("user_001", "oculus_quest")
        
        test_object = VirtualTestObject(
            object_id="vr_test_obj",
            test_id="vr_test_001",
            object_type="unit_test",
            position=(0, 0, 0),
            scale=(1, 1, 1),
            properties={},
            interactions=[],
            state="ready",
            metadata={}
        )
        
        result = self.vr_engine.execute_vr_test(session_id, test_object)
        
        self.assertIsInstance(result, dict)
        self.assertIn("test_id", result)
        self.assertIn("vr_metrics", result)
        self.assertIn("interactions", result)
        self.assertIn("success", result)
    
    def test_ar_device_initialization(self):
        """Test AR device initialization"""
        device_id = self.ar_overlay.initialize_ar_device("device_001", "hololens")
        
        self.assertEqual(device_id, "device_001")
        self.assertIn(device_id, self.ar_overlay.ar_devices)
        
        device = self.ar_overlay.ar_devices[device_id]
        self.assertEqual(device["device_type"], "hololens")
        self.assertTrue(device["camera_feed"])
    
    def test_ar_device_configurations(self):
        """Test AR device configurations"""
        device_types = ["hololens", "magic_leap", "iphone_ar", "android_ar"]
        
        for device_type in device_types:
            config = self.ar_overlay._get_ar_device_config(device_type)
            
            self.assertIsInstance(config, dict)
            self.assertIn("resolution", config)
            self.assertIn("fov", config)
            self.assertIn("tracking", config)
    
    def test_ar_overlay_creation(self):
        """Test AR overlay creation"""
        device_id = self.ar_overlay.initialize_ar_device("device_001", "hololens")
        
        test_data = {"test_name": "AR Test", "status": "running"}
        overlay_id = self.ar_overlay.create_test_overlay(device_id, test_data)
        
        self.assertIsNotNone(overlay_id)
        self.assertIn(overlay_id, self.ar_overlay.overlay_layers)
        
        overlay = self.ar_overlay.overlay_layers[overlay_id]
        self.assertEqual(overlay["device_id"], device_id)
        self.assertEqual(overlay["test_data"], test_data)
    
    def test_ar_overlay_anchoring(self):
        """Test AR overlay anchoring"""
        device_id = self.ar_overlay.initialize_ar_device("device_001", "hololens")
        overlay_id = self.ar_overlay.create_test_overlay(device_id, {"test": "data"})
        
        anchor_position = (1.0, 2.0, 3.0)
        success = self.ar_overlay.anchor_overlay_to_world(overlay_id, anchor_position)
        
        self.assertTrue(success)
        
        overlay = self.ar_overlay.overlay_layers[overlay_id]
        self.assertEqual(overlay["anchor_point"], anchor_position)
        self.assertEqual(overlay["position"], anchor_position)
    
    def test_collaboration_room_creation(self):
        """Test collaboration room creation"""
        room_id = self.collaboration.create_collaboration_room("Test Room", "user_001")
        
        self.assertIsNotNone(room_id)
        self.assertIn(room_id, self.collaboration.collaboration_rooms)
        
        room = self.collaboration.collaboration_rooms[room_id]
        self.assertEqual(room["room_name"], "Test Room")
        self.assertEqual(room["creator_id"], "user_001")
        self.assertIn("user_001", room["participants"])
    
    def test_collaboration_room_joining(self):
        """Test collaboration room joining"""
        room_id = self.collaboration.create_collaboration_room("Test Room", "user_001")
        
        success = self.collaboration.join_collaboration_room(room_id, "user_002")
        
        self.assertTrue(success)
        
        room = self.collaboration.collaboration_rooms[room_id]
        self.assertIn("user_002", room["participants"])
    
    def test_test_object_sharing(self):
        """Test test object sharing"""
        room_id = self.collaboration.create_collaboration_room("Test Room", "user_001")
        self.collaboration.join_collaboration_room(room_id, "user_002")
        
        test_object = VirtualTestObject(
            object_id="shared_obj",
            test_id="shared_test",
            object_type="integration_test",
            position=(0, 0, 0),
            scale=(1, 1, 1),
            properties={},
            interactions=[],
            state="ready",
            metadata={}
        )
        
        shared_object_id = self.collaboration.share_test_object(room_id, "user_001", test_object)
        
        self.assertIsNotNone(shared_object_id)
        self.assertIn(shared_object_id, self.collaboration.shared_objects)
        
        shared_object = self.collaboration.shared_objects[shared_object_id]
        self.assertEqual(shared_object["owner_id"], "user_001")
        self.assertEqual(shared_object["room_id"], room_id)
    
    def test_collaborative_test_action(self):
        """Test collaborative test action"""
        room_id = self.collaboration.create_collaboration_room("Test Room", "user_001")
        self.collaboration.join_collaboration_room(room_id, "user_002")
        
        test_object = VirtualTestObject(
            object_id="collab_obj",
            test_id="collab_test",
            object_type="unit_test",
            position=(0, 0, 0),
            scale=(1, 1, 1),
            properties={},
            interactions=[],
            state="ready",
            metadata={}
        )
        
        shared_object_id = self.collaboration.share_test_object(room_id, "user_001", test_object)
        
        action = self.collaboration.collaborate_on_test(room_id, "user_002", shared_object_id, "modify")
        
        self.assertIsInstance(action, dict)
        self.assertEqual(action["user_id"], "user_002")
        self.assertEqual(action["action"], "modify")
        self.assertIn("timestamp", action)
    
    def test_voice_channel_setup(self):
        """Test voice channel setup"""
        room_id = self.collaboration.create_collaboration_room("Test Room", "user_001")
        
        voice_channel_id = self.collaboration.setup_voice_channel(room_id)
        
        self.assertIsNotNone(voice_channel_id)
        self.assertIn(voice_channel_id, self.collaboration.voice_channels)
        
        voice_channel = self.collaboration.voice_channels[voice_channel_id]
        self.assertEqual(voice_channel["room_id"], room_id)
        self.assertTrue(voice_channel["spatial_audio"])

def run_metaverse_tests():
    """Run all metaverse tests"""
    logger.info("Running metaverse tests")
    
    # Create test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(MetaverseTestGenerator))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Log results
    logger.info(f"Metaverse tests completed: {result.testsRun} tests run")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    
    return result

if __name__ == "__main__":
    run_metaverse_tests()


