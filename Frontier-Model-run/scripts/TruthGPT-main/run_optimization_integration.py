"""
Main script to run the complete optimization integration.
"""

import sys
import os
import subprocess

def run_optimization_tests():
    """Run optimization tests."""
    print("🧪 Running optimization tests...")
    
    try:
        result = subprocess.run([
            sys.executable, "test_optimizations.py"
        ], capture_output=True, text=True, cwd="/home/ubuntu/TruthGPT")
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False

def apply_optimizations():
    """Apply optimizations to all variants."""
    print("\n🔧 Applying optimizations...")
    
    try:
        result = subprocess.run([
            sys.executable, "apply_optimizations.py"
        ], capture_output=True, text=True, cwd="/home/ubuntu/TruthGPT")
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Optimization application failed: {e}")
        return False

def main():
    """Main integration function."""
    print("🚀 TruthGPT Optimization Integration")
    print("=" * 60)
    
    success = True
    
    if run_optimization_tests():
        print("✅ Optimization tests passed")
    else:
        print("❌ Optimization tests failed")
        success = False
    
    if apply_optimizations():
        print("✅ Optimizations applied successfully")
    else:
        print("❌ Optimization application failed")
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Optimization integration completed successfully!")
        print("📊 Check OPTIMIZATION_REPORT.md for detailed results")
    else:
        print("⚠️  Optimization integration completed with issues")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
