
import sys
import os

# Add parent of optimization_core to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

print(f"Sys Path: {sys.path[0]}")

try:
    print("Attempting to import optimization_core.modules.feed_forward...")
    import optimization_core.modules.feed_forward as ff
    print(f"Feed Forward module: {ff}")
    
    print("Attempting to access ProductionPiMoESystem from feed_forward...")
    prod_sys = ff.ProductionPiMoESystem
    print(f"Imported: {prod_sys}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

