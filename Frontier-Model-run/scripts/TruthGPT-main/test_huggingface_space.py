"""
Test script for the updated Hugging Face Space with RL-enhanced optimization.
"""

import sys
import os
sys.path.append('.')
sys.path.append('./huggingface_space')

def test_imports():
    """Test that all required imports work correctly."""
    print("🧪 Testing Hugging Face Space imports...")
    
    try:
        from huggingface_space.app import TruthGPTDemo, create_gradio_interface
        print("✅ Main app imports successful")
        
        from optimization_core.hybrid_optimization_core import create_hybrid_optimization_core
        print("✅ Hybrid optimization core import successful")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_demo_initialization():
    """Test that the TruthGPTDemo class initializes correctly."""
    print("\n🧪 Testing TruthGPTDemo initialization...")
    
    try:
        from huggingface_space.app import TruthGPTDemo
        demo = TruthGPTDemo()
        
        print(f"✅ Demo initialized successfully")
        print(f"   Models loaded: {len(demo.models)}")
        print(f"   Hybrid optimizer available: {demo.hybrid_optimizer is not None}")
        
        if demo.hybrid_optimizer:
            print("✅ RL-enhanced optimization core loaded")
        else:
            print("⚠️ RL optimization running in demo mode")
        
        return True
    except Exception as e:
        print(f"❌ Demo initialization error: {e}")
        return False

def test_rl_optimization_demo():
    """Test the RL optimization demo functionality."""
    print("\n🧪 Testing RL optimization demo...")
    
    try:
        from huggingface_space.app import TruthGPTDemo
        demo = TruthGPTDemo()
        
        if not demo.models:
            print("⚠️ No models available for testing")
            return True
        
        model_name = list(demo.models.keys())[0]
        result = demo.run_rl_optimization_demo(model_name)
        
        print(f"✅ RL demo executed for {model_name}")
        print(f"   Result length: {len(result)} characters")
        
        if "DAPO" in result and "VAPO" in result and "ORZ" in result:
            print("✅ All RL techniques mentioned in output")
        else:
            print("⚠️ Some RL techniques missing from output")
        
        return True
    except Exception as e:
        print(f"❌ RL demo test error: {e}")
        return False

def test_gradio_interface():
    """Test that the Gradio interface can be created."""
    print("\n🧪 Testing Gradio interface creation...")
    
    try:
        from huggingface_space.app import create_gradio_interface
        interface = create_gradio_interface()
        
        print("✅ Gradio interface created successfully")
        print(f"   Interface type: {type(interface)}")
        
        return True
    except Exception as e:
        print(f"❌ Gradio interface test error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Hugging Face Space Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_demo_initialization,
        test_rl_optimization_demo,
        test_gradio_interface
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Hugging Face Space is ready.")
    else:
        print("⚠️ Some tests failed. Check the output above.")
