import sys
import os
import importlib

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PARENT_ROOT = os.path.dirname(PROJECT_ROOT)

print(f"CWD: {os.getcwd()}")
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"PARENT_ROOT: {PARENT_ROOT}")

if PARENT_ROOT not in sys.path:
    print(f"Adding {PARENT_ROOT} to sys.path")
    sys.path.insert(0, PARENT_ROOT)

print(f"sys.path: {sys.path[:3]}")

try:
    import optimization_core
    print(f"SUCCESS: optimization_core imported from {optimization_core.__file__}")
    
    from optimization_core.modules.enterprise.auth import EnterpriseAuth
    print(f"SUCCESS: EnterpriseAuth imported: {EnterpriseAuth}")
    
    from optimization_core.utils.gpu import CUDAOptimizations
    print(f"SUCCESS: CUDAOptimizations imported: {CUDAOptimizations}")
    
except Exception as e:
    print(f"FAILURE: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
