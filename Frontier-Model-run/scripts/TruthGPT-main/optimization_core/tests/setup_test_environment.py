"""
Setup script for TruthGPT optimization core test environment
Configures testing environment and validates setup
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path
from typing import List, Dict, Any, Optional

class TestEnvironmentSetup:
    """Setup and validate test environment"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_root = Path(__file__).parent
        self.required_packages = [
            'torch',
            'numpy',
            'psutil',
            'pytest',
            'unittest',
            'json',
            'time',
            'gc'
        ]
        self.optional_packages = [
            'pytest-cov',
            'pytest-xdist',
            'pytest-benchmark',
            'memory-profiler',
            'line-profiler'
        ]
        
    def check_python_version(self) -> bool:
        """Check Python version compatibility"""
        print("🐍 Checking Python version...")
        
        if sys.version_info < (3, 8):
            print(f"❌ Python {sys.version} is not supported. Requires Python 3.8+")
            return False
        
        print(f"✅ Python {sys.version} is compatible")
        return True
    
    def check_required_packages(self) -> Dict[str, bool]:
        """Check if required packages are installed"""
        print("📦 Checking required packages...")
        
        package_status = {}
        
        for package in self.required_packages:
            try:
                importlib.import_module(package)
                package_status[package] = True
                print(f"✅ {package} is installed")
            except ImportError:
                package_status[package] = False
                print(f"❌ {package} is not installed")
        
        return package_status
    
    def check_optional_packages(self) -> Dict[str, bool]:
        """Check if optional packages are installed"""
        print("🔧 Checking optional packages...")
        
        package_status = {}
        
        for package in self.optional_packages:
            try:
                importlib.import_module(package.replace('-', '_'))
                package_status[package] = True
                print(f"✅ {package} is installed")
            except ImportError:
                package_status[package] = False
                print(f"⚠️  {package} is not installed (optional)")
        
        return package_status
    
    def check_torch_installation(self) -> Dict[str, Any]:
        """Check PyTorch installation and capabilities"""
        print("🔥 Checking PyTorch installation...")
        
        try:
            import torch
            torch_info = {
                'version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            }
            
            print(f"✅ PyTorch {torch_info['version']} is installed")
            if torch_info['cuda_available']:
                print(f"✅ CUDA is available (version: {torch_info['cuda_version']})")
                print(f"✅ {torch_info['device_count']} CUDA devices available")
            if torch_info['mps_available']:
                print("✅ MPS (Metal Performance Shaders) is available")
            
            return torch_info
            
        except ImportError:
            print("❌ PyTorch is not installed")
            return {}
    
    def check_test_structure(self) -> bool:
        """Check if test structure is properly set up"""
        print("📁 Checking test structure...")
        
        required_dirs = [
            'tests',
            'tests/unit',
            'tests/integration', 
            'tests/performance',
            'tests/fixtures'
        ]
        
        required_files = [
            'tests/__init__.py',
            'tests/conftest.py',
            'tests/run_all_tests.py',
            'tests/fixtures/__init__.py',
            'tests/fixtures/test_data.py',
            'tests/fixtures/mock_components.py',
            'tests/fixtures/test_utils.py'
        ]
        
        all_good = True
        
        for dir_path in required_dirs:
            full_path = self.test_root / dir_path
            if full_path.exists():
                print(f"✅ {dir_path} exists")
            else:
                print(f"❌ {dir_path} is missing")
                all_good = False
        
        for file_path in required_files:
            full_path = self.test_root / file_path
            if full_path.exists():
                print(f"✅ {file_path} exists")
            else:
                print(f"❌ {file_path} is missing")
                all_good = False
        
        return all_good
    
    def check_test_imports(self) -> bool:
        """Check if test modules can be imported"""
        print("🔍 Checking test imports...")
        
        test_modules = [
            'tests.fixtures.test_data',
            'tests.fixtures.mock_components',
            'tests.fixtures.test_utils'
        ]
        
        all_good = True
        
        for module in test_modules:
            try:
                importlib.import_module(module)
                print(f"✅ {module} imports successfully")
            except ImportError as e:
                print(f"❌ {module} import failed: {e}")
                all_good = False
        
        return all_good
    
    def run_basic_tests(self) -> bool:
        """Run basic tests to verify setup"""
        print("🧪 Running basic tests...")
        
        try:
            # Test basic functionality
            from tests.fixtures.test_data import TestDataFactory
            from tests.fixtures.mock_components import MockModel, MockOptimizer
            
            # Create test instances
            factory = TestDataFactory()
            model = MockModel()
            optimizer = MockOptimizer()
            
            # Test basic operations
            data = factory.create_mlp_data()
            output = model(data)
            result = optimizer.step(torch.tensor(1.0))
            
            print("✅ Basic tests passed")
            return True
            
        except Exception as e:
            print(f"❌ Basic tests failed: {e}")
            return False
    
    def install_missing_packages(self, missing_packages: List[str]) -> bool:
        """Install missing packages"""
        if not missing_packages:
            return True
        
        print(f"📦 Installing missing packages: {missing_packages}")
        
        try:
            for package in missing_packages:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✅ {package} installed successfully")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install packages: {e}")
            return False
    
    def create_test_config(self) -> bool:
        """Create test configuration file"""
        print("⚙️  Creating test configuration...")
        
        config_content = """# TruthGPT Optimization Core Test Configuration

# Test execution settings
VERBOSE = True
PARALLEL = False
COVERAGE = True
PERFORMANCE = True
INTEGRATION = True

# Performance settings
BENCHMARK_ITERATIONS = 100
BENCHMARK_WARMUP = 10
BENCHMARK_TIMEOUT = 300

# Memory settings
MEMORY_LIMIT_MB = 1024
MEMORY_WARNING_MB = 512

# Test data settings
DEFAULT_BATCH_SIZE = 2
DEFAULT_SEQ_LEN = 128
DEFAULT_D_MODEL = 512

# Device settings
PREFER_CUDA = True
PREFER_MPS = False
FALLBACK_CPU = True
"""
        
        config_file = self.test_root / "test_config.py"
        
        try:
            with open(config_file, 'w') as f:
                f.write(config_content)
            print(f"✅ Test configuration created: {config_file}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create test configuration: {e}")
            return False
    
    def validate_environment(self) -> Dict[str, Any]:
        """Validate complete test environment"""
        print("🔍 Validating test environment...")
        
        validation_results = {
            'python_version': self.check_python_version(),
            'required_packages': self.check_required_packages(),
            'optional_packages': self.check_optional_packages(),
            'torch_info': self.check_torch_installation(),
            'test_structure': self.check_test_structure(),
            'test_imports': self.check_test_imports(),
            'basic_tests': self.run_basic_tests()
        }
        
        # Check if all required packages are installed
        missing_required = [pkg for pkg, status in validation_results['required_packages'].items() if not status]
        
        if missing_required:
            print(f"❌ Missing required packages: {missing_required}")
            validation_results['missing_packages'] = missing_required
        else:
            validation_results['missing_packages'] = []
        
        return validation_results
    
    def setup_environment(self) -> bool:
        """Complete environment setup"""
        print("🚀 Setting up TruthGPT test environment...")
        print("=" * 60)
        
        # Validate environment
        validation_results = self.validate_environment()
        
        # Install missing packages if any
        if validation_results['missing_packages']:
            print(f"\n📦 Installing missing packages...")
            if not self.install_missing_packages(validation_results['missing_packages']):
                print("❌ Failed to install missing packages")
                return False
        
        # Create test configuration
        if not self.create_test_config():
            print("❌ Failed to create test configuration")
            return False
        
        # Final validation
        print("\n🔍 Final validation...")
        final_validation = self.validate_environment()
        
        # Check if everything is working
        all_good = (
            final_validation['python_version'] and
            all(final_validation['required_packages'].values()) and
            final_validation['test_structure'] and
            final_validation['test_imports'] and
            final_validation['basic_tests']
        )
        
        if all_good:
            print("\n🎉 Test environment setup completed successfully!")
            print("✅ All required components are working")
            print("✅ Test structure is properly set up")
            print("✅ Test imports are working")
            print("✅ Basic tests are passing")
            
            # Print optional package status
            optional_status = final_validation['optional_packages']
            installed_optional = [pkg for pkg, status in optional_status.items() if status]
            if installed_optional:
                print(f"✅ Optional packages installed: {installed_optional}")
            
            # Print PyTorch info
            torch_info = final_validation['torch_info']
            if torch_info:
                print(f"✅ PyTorch {torch_info['version']} is working")
                if torch_info['cuda_available']:
                    print(f"✅ CUDA is available with {torch_info['device_count']} devices")
                if torch_info['mps_available']:
                    print("✅ MPS is available")
            
            return True
        else:
            print("\n❌ Test environment setup failed!")
            print("Please check the errors above and fix them before running tests")
            return False
    
    def print_setup_summary(self, validation_results: Dict[str, Any]):
        """Print setup summary"""
        print("\n📊 Setup Summary:")
        print("=" * 40)
        
        # Python version
        print(f"Python Version: {'✅' if validation_results['python_version'] else '❌'}")
        
        # Required packages
        required_status = validation_results['required_packages']
        installed_required = sum(1 for status in required_status.values() if status)
        total_required = len(required_status)
        print(f"Required Packages: {installed_required}/{total_required} {'✅' if installed_required == total_required else '❌'}")
        
        # Optional packages
        optional_status = validation_results['optional_packages']
        installed_optional = sum(1 for status in optional_status.values() if status)
        total_optional = len(optional_status)
        print(f"Optional Packages: {installed_optional}/{total_optional}")
        
        # Test structure
        print(f"Test Structure: {'✅' if validation_results['test_structure'] else '❌'}")
        
        # Test imports
        print(f"Test Imports: {'✅' if validation_results['test_imports'] else '❌'}")
        
        # Basic tests
        print(f"Basic Tests: {'✅' if validation_results['basic_tests'] else '❌'}")
        
        # PyTorch info
        torch_info = validation_results['torch_info']
        if torch_info:
            print(f"PyTorch: ✅ {torch_info['version']}")
            if torch_info['cuda_available']:
                print(f"CUDA: ✅ {torch_info['device_count']} devices")
            if torch_info['mps_available']:
                print("MPS: ✅ Available")

def main():
    """Main setup function"""
    setup = TestEnvironmentSetup()
    
    print("🔧 TruthGPT Optimization Core Test Environment Setup")
    print("=" * 60)
    
    # Run setup
    success = setup.setup_environment()
    
    if success:
        print("\n🎉 Setup completed successfully!")
        print("You can now run tests with:")
        print("  python tests/run_all_tests.py")
        print("  pytest tests/")
        sys.exit(0)
    else:
        print("\n❌ Setup failed!")
        print("Please fix the issues above and run setup again")
        sys.exit(1)

if __name__ == "__main__":
    main()





