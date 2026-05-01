import sys
import os
import traceback

print(f"Python Version: {sys.version}")

try:
    import web3
    print(f"SUCCESS: web3 version {web3.__version__}")
    from web3 import Web3
    
    try:
        from web3.middleware import geth_poa_middleware
        print("SUCCESS: geth_poa_middleware imported from web3.middleware")
    except ImportError:
        print("FAILURE: geth_poa_middleware NOT found in web3.middleware")
        try:
            # In newer web3 versions it might be elsewhere
            from web3.middleware import ExtraDataToPOAMiddleware
            print("SUCCESS: ExtraDataToPOAMiddleware found (v6+ style)")
        except ImportError:
            print("FAILURE: ExtraDataToPOAMiddleware NOT found")

except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")
    traceback.print_exc()
