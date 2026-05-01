import sys
import importlib.util
import shutil

def check_python_version():
    """Verify Python version is 3.10+"""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 10):
        print(f"❌ Python 3.10+ is required. Found: {sys.version.split()[0]}")
        return False
    print(f"✅ Python Version: {sys.version.split()[0]}")
    return True

def check_package(package_name, alias=None):
    """Check if a package is importable."""
    name = alias if alias else package_name
    if importlib.util.find_spec(package_name) is None:
        print(f"❌ Missing package: {name}")
        return False
    
    # Optional: Print version if possible
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "unknown")
        print(f"✅ Found {name}: {version}")
    except ImportError:
        print(f"❌ Failed to import: {name}")
        return False
    return True

def check_cuda():
    """Check PyTorch CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA Available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  CUDA Not Available (Running on CPU)")
            return True # Not a hard failure, but worth noting
    except ImportError:
        print("❌ PyTorch not found (CUDA check skipped)")
        return False

def main():
    print("\n🔎 Pre-flight System Check...\n")
    
    checks = [
        check_python_version(),
        check_package("torch", "PyTorch"),
        check_package("transformers", "Transformers"),
        check_cuda()
    ]
    
    if all(checks):
        print("\n🚀 System Ready! TruthGPT Optimization Core is correctly installed.")
        sys.exit(0)
    else:
        print("\n❌ Some checks failed. Please review the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

