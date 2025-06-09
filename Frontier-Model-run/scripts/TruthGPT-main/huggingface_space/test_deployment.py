"""
Test script for Hugging Face Space deployment functionality
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_export():
    """Test model export functionality."""
    print("🧪 Testing model export...")
    
    try:
        from model_export import ModelExporter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = ModelExporter(temp_dir)
            exported = exporter.export_mock_models()
            
            assert len(exported) > 0, "No models were exported"
            
            for name, path in exported.items():
                model_dir = Path(path)
                assert model_dir.exists(), f"Model directory {path} does not exist"
                assert (model_dir / "pytorch_model.bin").exists(), f"Model file missing for {name}"
                assert (model_dir / "config.json").exists(), f"Config file missing for {name}"
                assert (model_dir / "README.md").exists(), f"README missing for {name}"
            
            print(f"✅ Model export test passed! Exported {len(exported)} models")
            return True
            
    except Exception as e:
        print(f"❌ Model export test failed: {e}")
        return False

def test_gradio_app():
    """Test Gradio app initialization."""
    print("🧪 Testing Gradio app...")
    
    try:
        from app import TruthGPTDemo, create_gradio_interface
        
        demo_instance = TruthGPTDemo()
        assert len(demo_instance.models) > 0, "No models loaded"
        
        interface = create_gradio_interface()
        assert interface is not None, "Failed to create Gradio interface"
        
        print("✅ Gradio app test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Gradio app test failed: {e}")
        return False

def test_deployment_script():
    """Test deployment script functionality."""
    print("🧪 Testing deployment script...")
    
    try:
        from deploy import HuggingFaceSpaceDeployer
        
        deployer = HuggingFaceSpaceDeployer("test-space", "test-user")
        assert deployer.space_name == "test-space", "Space name not set correctly"
        assert deployer.username == "test-user", "Username not set correctly"
        
        print("✅ Deployment script test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Deployment script test failed: {e}")
        return False

def main():
    """Run all deployment tests."""
    print("🚀 Running Hugging Face Space Deployment Tests")
    print("=" * 60)
    
    tests = [
        test_model_export,
        test_gradio_app,
        test_deployment_script
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All deployment tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
