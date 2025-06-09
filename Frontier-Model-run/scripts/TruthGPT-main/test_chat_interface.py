#!/usr/bin/env python3
"""
Test script to verify the chat interface functionality works correctly.
"""

import sys
import os
sys.path.append('./huggingface_space')

def test_chat_interface():
    """Test that the chat interface works correctly."""
    print('🧪 Testing Chat Interface functionality...')
    
    try:
        from huggingface_space.app import TruthGPTDemo
        
        demo = TruthGPTDemo()
        
        print(f'Models available: {list(demo.models.keys())}')
        print(f'Chat method available: {hasattr(demo, "chat_with_model")}')
        
        if hasattr(demo, 'chat_with_model'):
            print('✅ Testing chat with DeepSeek-V3-Demo...')
            
            test_message = "Hello, how are you?"
            initial_history = []
            
            result_history, cleared_input = demo.chat_with_model(
                'DeepSeek-V3-Demo', 
                test_message, 
                initial_history
            )
            
            print('Chat result preview:')
            if result_history and len(result_history) > 0:
                last_exchange = result_history[-1]
                print(f'User: {last_exchange[0]}')
                print(f'Model: {last_exchange[1][:200]}...')
                print(f'✅ Chat working successfully!')
                return True
            else:
                print('❌ Chat returned empty history')
                return False
        else:
            print('❌ Chat method not found')
            return False
            
    except Exception as e:
        print(f'❌ Test failed with error: {e}')
        return False

if __name__ == "__main__":
    success = test_chat_interface()
    if success:
        print('\n🎉 Chat interface test passed!')
    else:
        print('\n❌ Chat interface test failed!')
