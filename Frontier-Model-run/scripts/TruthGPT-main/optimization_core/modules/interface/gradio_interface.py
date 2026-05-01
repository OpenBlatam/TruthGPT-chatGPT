"""
Ultra-fast modular Gradio interface
Following deep learning best practices
"""

import gradio as gr
import torch
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
import json
import time

from ..base.logger import BaseLogger


@dataclass
class GradioConfig:
    """Gradio interface configuration"""
    title: str = "TruthGPT Ultra-Fast Interface"
    description: str = "Advanced LLM optimization and interaction"
    theme: str = "soft"
    server_name: str = "127.0.0.1"
    server_port: int = 7860
    share: bool = False
    debug: bool = False


class FastGradioInterface:
    """Ultra-fast Gradio interface"""
    
    def __init__(self, model: torch.nn.Module, config: GradioConfig, logger: Optional[BaseLogger] = None):
        self.model = model
        self.config = config
        self.logger = logger or BaseLogger("FastGradioInterface")
        
        # Interface state
        self.chat_history = []
        self.generation_params = {
            'temperature': 1.0,
            'top_p': 0.9,
            'max_length': 100,
            'do_sample': True
        }
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""
        with gr.Blocks(
            title=self.config.title,
            theme=gr.themes.Soft(),
            css=self._get_css()
        ) as interface:
            
            gr.Markdown(f"# {self.config.title}")
            gr.Markdown(self.config.description)
            
            with gr.Tabs():
                # Chat Tab
                with gr.Tab("💬 Chat"):
                    self._create_chat_tab()
                
                # Generation Tab
                with gr.Tab("📝 Generation"):
                    self._create_generation_tab()
                
                # Settings Tab
                with gr.Tab("⚙️ Settings"):
                    self._create_settings_tab()
            
            return interface
    
    def _create_chat_tab(self):
        """Create chat interface"""
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Chat History", height=400)
                msg_input = gr.Textbox(label="Your Message", placeholder="Type here...", lines=2)
                
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear", variant="secondary")
            
            with gr.Column(scale=1):
                gr.Markdown("### Parameters")
                temperature = gr.Slider(0.1, 2.0, 1.0, step=0.1, label="Temperature")
                max_length = gr.Slider(10, 500, 100, step=10, label="Max Length")
                top_p = gr.Slider(0.1, 1.0, 0.9, step=0.05, label="Top-p")
                do_sample = gr.Checkbox(label="Use Sampling", value=True)
        
        # Event handlers
        def chat_response(message, history, temp, max_len, top_p_val, sample):
            if not message.strip():
                return history, ""
            
            try:
                # Generate response
                response = self._generate_response(message, temp, max_len, top_p_val, sample)
                history.append([message, response])
                return history, ""
            except Exception as e:
                self.logger.error(f"Chat error: {e}")
                return history, f"Error: {str(e)}"
        
        def clear_chat():
            return [], ""
        
        send_btn.click(chat_response, [msg_input, chatbot, temperature, max_length, top_p, do_sample], [chatbot, msg_input])
        msg_input.submit(chat_response, [msg_input, chatbot, temperature, max_length, top_p, do_sample], [chatbot, msg_input])
        clear_btn.click(clear_chat, outputs=[chatbot, msg_input])
    
    def _create_generation_tab(self):
        """Create text generation interface"""
        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(label="Input Text", placeholder="Enter prompt...", lines=3)
                
                with gr.Row():
                    generate_btn = gr.Button("Generate", variant="primary")
                    clear_btn = gr.Button("Clear", variant="secondary")
                
                gr.Markdown("### Generation Parameters")
                with gr.Row():
                    gen_temperature = gr.Slider(0.1, 2.0, 1.0, step=0.1, label="Temperature")
                    gen_max_length = gr.Slider(10, 1000, 200, step=10, label="Max Length")
                
                with gr.Row():
                    gen_top_p = gr.Slider(0.1, 1.0, 0.9, step=0.05, label="Top-p")
                    gen_top_k = gr.Slider(1, 100, 50, step=1, label="Top-k")
                
                gen_do_sample = gr.Checkbox(label="Use Sampling", value=True)
            
            with gr.Column(scale=1):
                output_text = gr.Textbox(label="Generated Text", lines=15, interactive=False)
                
                with gr.Row():
                    copy_btn = gr.Button("Copy", variant="secondary")
                    save_btn = gr.Button("Save", variant="secondary")
        
        def generate_text(prompt, temp, max_len, top_p_val, top_k_val, sample):
            if not prompt.strip():
                return "Please enter a prompt."
            
            try:
                return self._generate_response(prompt, temp, max_len, top_p_val, sample)
            except Exception as e:
                self.logger.error(f"Generation error: {e}")
                return f"Error: {str(e)}"
        
        def clear_generation():
            return "", ""
        
        generate_btn.click(generate_text, [input_text, gen_temperature, gen_max_length, gen_top_p, gen_top_k, gen_do_sample], [output_text])
        clear_btn.click(clear_generation, outputs=[input_text, output_text])
        copy_btn.click(lambda x: x, inputs=[output_text], outputs=gr.Textbox(label="Copied"))
        save_btn.click(lambda x: f"Saved: {x[:50]}..." if x else "No text to save", inputs=[output_text], outputs=gr.Textbox(label="Save Status"))
    
    def _create_settings_tab(self):
        """Create settings interface"""
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Model Settings")
                
                model_name = gr.Textbox(label="Model Name", value="TruthGPT", interactive=True)
                max_length = gr.Number(label="Max Sequence Length", value=2048, precision=0)
                hidden_size = gr.Number(label="Hidden Size", value=768, precision=0)
                num_heads = gr.Number(label="Attention Heads", value=12, precision=0)
                
                gr.Markdown("### Optimization Settings")
                
                use_mixed_precision = gr.Checkbox(label="Mixed Precision", value=True)
                use_gradient_checkpointing = gr.Checkbox(label="Gradient Checkpointing", value=True)
                use_flash_attention = gr.Checkbox(label="Flash Attention", value=True)
                use_lora = gr.Checkbox(label="LoRA Fine-tuning", value=False)
                
                with gr.Row():
                    save_settings_btn = gr.Button("Save Settings", variant="primary")
                    reset_settings_btn = gr.Button("Reset", variant="secondary")
            
            with gr.Column(scale=1):
                gr.Markdown("### System Info")
                
                system_info = gr.Textbox(
                    label="System Information",
                    value=self._get_system_info(),
                    lines=10,
                    interactive=False
                )
        
        def save_settings(model_name, max_len, hidden_size, num_heads, mixed_precision, 
                         gradient_checkpointing, flash_attention, lora):
            try:
                # Update settings
                self.generation_params.update({
                    'model_name': model_name,
                    'max_length': max_len,
                    'hidden_size': hidden_size,
                    'num_heads': num_heads,
                    'use_mixed_precision': mixed_precision,
                    'use_gradient_checkpointing': gradient_checkpointing,
                    'use_flash_attention': flash_attention,
                    'use_lora': lora
                })
                return "Settings saved successfully!"
            except Exception as e:
                return f"Error saving settings: {str(e)}"
        
        def reset_settings():
            return "TruthGPT", 2048, 768, 12, True, True, True, False
        
        save_settings_btn.click(save_settings, [model_name, max_length, hidden_size, num_heads, 
                                             use_mixed_precision, use_gradient_checkpointing, 
                                             use_flash_attention, use_lora], 
                               outputs=gr.Textbox(label="Save Status"))
        reset_settings_btn.click(reset_settings, outputs=[model_name, max_length, hidden_size, num_heads, 
                                                         use_mixed_precision, use_gradient_checkpointing, 
                                                         use_flash_attention, use_lora])
    
    def _generate_response(self, text: str, temperature: float, max_length: int, 
                         top_p: float, do_sample: bool) -> str:
        """Generate response from model"""
        try:
            # Tokenize input
            input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # Placeholder tokenization
            
            # Generate
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample
                )
            
            # Decode (placeholder)
            return f"Generated response for: {text[:50]}..."
            
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            raise
    
    def _get_system_info(self) -> str:
        """Get system information"""
        return f"""
        Model: {self.model.__class__.__name__}
        Device: {next(self.model.parameters()).device}
        Parameters: {sum(p.numel() for p in self.model.parameters()):,}
        Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB (if CUDA)
        """
    
    def _get_css(self) -> str:
        """Get custom CSS"""
        return """
        .gradio-container {
            max-width: 1200px !important;
        }
        .chat-message {
            padding: 10px;
            margin: 5px 0;
            border-radius: 10px;
        }
        .user-message {
            background-color: #e3f2fd;
            text-align: right;
        }
        .bot-message {
            background-color: #f3e5f5;
            text-align: left;
        }
        """
    
    def launch(self, **kwargs):
        """Launch the interface"""
        try:
            interface = self.create_interface()
            interface.launch(
                server_name=self.config.server_name,
                server_port=self.config.server_port,
                share=self.config.share,
                **kwargs
            )
        except Exception as e:
            self.logger.error(f"Error launching interface: {e}")
            raise


def create_interface(model: torch.nn.Module, config: Optional[GradioConfig] = None) -> FastGradioInterface:
    """Create Gradio interface"""
    if config is None:
        config = GradioConfig()
    
    return FastGradioInterface(model, config)



