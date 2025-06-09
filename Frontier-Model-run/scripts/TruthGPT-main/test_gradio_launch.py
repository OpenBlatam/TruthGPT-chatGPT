#!/usr/bin/env python3
"""
Test script to launch the Gradio app and verify chat interface functionality.
"""

import sys
import os
sys.path.append('./huggingface_space')

def launch_gradio_app():
    """Launch the Gradio app for testing."""
    print('🚀 Launching TruthGPT Gradio Space with Chat Interface...')
    
    try:
        from huggingface_space.app import create_gradio_interface
        
        interface = create_gradio_interface()
        
        print('✅ Gradio interface created successfully')
        print('🌐 Starting server on port 7864...')
        
        interface.launch(
            server_port=7864,
            share=False,
            quiet=False,
            show_error=True
        )
        
    except Exception as e:
        print(f'❌ Failed to launch Gradio app: {e}')
        return False

if __name__ == "__main__":
    launch_gradio_app()
