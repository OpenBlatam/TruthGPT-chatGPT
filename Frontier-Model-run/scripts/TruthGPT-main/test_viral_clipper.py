#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import torch
from variant.viral_clipper import create_viral_clipper_model
from variant.video_processor import VideoProcessor

def test_model_instantiation():
    """Test that the viral clipper model can be instantiated."""
    print("Testing viral clipper model instantiation...")
    
    config = {
        'hidden_size': 256,
        'num_layers': 3,
        'num_attention_heads': 4,
        'dropout': 0.1,
        'max_sequence_length': 50,
        'max_duration': 300,
        'clip_duration': 30,
        'min_clip_duration': 10,
        'max_clip_duration': 60,
        'engagement_threshold': 0.8,
        'view_velocity_threshold': 1000,
        'comment_ratio_threshold': 0.05,
        'like_ratio_threshold': 0.1,
        'visual_feature_dim': 512,
        'audio_feature_dim': 256,
        'text_feature_dim': 384,
        'engagement_feature_dim': 32
    }
    
    try:
        model = create_viral_clipper_model(config)
        print(f"✓ Model instantiated successfully")
        print(f"  - Model type: {type(model)}")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model
    except Exception as e:
        print(f"✗ Model instantiation failed: {e}")
        return None

def test_forward_pass(model):
    """Test that the model can perform a forward pass."""
    print("\nTesting forward pass...")
    
    try:
        batch_size = 2
        seq_len = 20
        
        visual_features = torch.randn(batch_size, seq_len, 512)
        audio_features = torch.randn(batch_size, seq_len, 256)
        text_features = torch.randn(batch_size, seq_len, 384)
        engagement_features = torch.randn(batch_size, seq_len, 32)
        
        with torch.no_grad():
            outputs = model(visual_features, audio_features, text_features, engagement_features)
        
        expected_virality_shape = (batch_size, seq_len, 1)
        expected_segment_shape = (batch_size, seq_len, 2)
        
        if (outputs['virality_scores'].shape == expected_virality_shape and 
            outputs['segment_logits'].shape == expected_segment_shape):
            print(f"✓ Forward pass successful")
            print(f"  - Virality scores shape: {outputs['virality_scores'].shape}")
            print(f"  - Segment logits shape: {outputs['segment_logits'].shape}")
            print(f"  - Attention layers: {len(outputs['attention_weights'])}")
            return True
        else:
            print(f"✗ Forward pass failed: unexpected output shapes")
            print(f"  - Expected virality: {expected_virality_shape}, got: {outputs['virality_scores'].shape}")
            print(f"  - Expected segment: {expected_segment_shape}, got: {outputs['segment_logits'].shape}")
            return False
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False

def test_viral_segment_prediction(model):
    """Test viral segment prediction functionality."""
    print("\nTesting viral segment prediction...")
    
    try:
        batch_size = 1
        seq_len = 20
        
        features = {
            'visual_features': torch.randn(batch_size, seq_len, 512),
            'audio_features': torch.randn(batch_size, seq_len, 256),
            'text_features': torch.randn(batch_size, seq_len, 384),
            'engagement_features': torch.randn(batch_size, seq_len, 32)
        }
        
        segments = model.predict_viral_segments(features, threshold=0.5)
        
        print(f"✓ Viral segment prediction successful")
        print(f"  - Predicted segments: {len(segments)}")
        if segments:
            print(f"  - Sample segment: {segments[0]}")
        return True
    except Exception as e:
        print(f"✗ Viral segment prediction failed: {e}")
        return False

def test_video_processor():
    """Test video processor functionality."""
    print("\nTesting video processor...")
    
    try:
        config = {'fps': 30, 'resolution': '720p'}
        processor = VideoProcessor(config)
        
        mock_metadata = {
            'title': 'Test Video',
            'duration': 300,
            'view_count': 1000000,
            'like_count': 50000,
            'comment_count': 5000
        }
        
        features = processor.process_video("mock_video.mp4", mock_metadata)
        
        required_keys = ['visual_features', 'audio_features', 'text_features', 'engagement_features']
        if all(key in features for key in required_keys):
            print(f"✓ Video processor test successful")
            print(f"  - Features extracted: {list(features.keys())}")
            print(f"  - Visual features shape: {features['visual_features'].shape}")
            print(f"  - Engagement features shape: {features['engagement_features'].shape}")
            return True
        else:
            print(f"✗ Video processor test failed: missing features")
            return False
    except Exception as e:
        print(f"✗ Video processor test failed: {e}")
        return False

def main():
    print("Viral Video Clipper Test")
    print("=" * 40)
    
    model = test_model_instantiation()
    if model is None:
        print("\nTest failed: Could not instantiate model")
        return False
    
    success = test_forward_pass(model)
    if not success:
        print("\nTest failed: Forward pass failed")
        return False
    
    success = test_viral_segment_prediction(model)
    if not success:
        print("\nTest failed: Viral segment prediction failed")
        return False
    
    success = test_video_processor()
    if not success:
        print("\nTest failed: Video processor failed")
        return False
    
    print("\n" + "=" * 40)
    print("✓ All tests passed! Viral clipper implementation is working.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
