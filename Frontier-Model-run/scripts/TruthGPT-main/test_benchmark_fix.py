"""
Test script to verify MockBenchmark functionality works correctly.
"""

import sys
import os
sys.path.append('.')
sys.path.append('./huggingface_space')

def test_mock_benchmark():
    """Test that MockBenchmark works correctly."""
    print('🧪 Testing MockBenchmark functionality...')
    
    try:
        from huggingface_space.app import TruthGPTDemo
        
        demo = TruthGPTDemo()
        
        print(f'Models available: {list(demo.models.keys())}')
        print(f'Benchmark available: {demo.benchmark is not None}')
        
        if demo.benchmark:
            print('✅ Testing benchmark for Llama-3.1-405B-Demo...')
            result = demo.run_benchmark('Llama-3.1-405B-Demo')
            
            print('Benchmark result preview:')
            print(result[:300] + '...' if len(result) > 300 else result)
            
            if '❌ Benchmark suite not available' in result:
                print('❌ Benchmark still not working')
                return False
            else:
                print('✅ Benchmark working successfully!')
                return True
        else:
            print('❌ Benchmark is None')
            return False
            
    except Exception as e:
        print(f'❌ Test failed with error: {e}')
        return False

if __name__ == "__main__":
    success = test_mock_benchmark()
    if success:
        print('\n🎉 MockBenchmark test passed!')
    else:
        print('\n❌ MockBenchmark test failed!')
