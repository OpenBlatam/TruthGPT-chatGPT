"""
Gradio Interface for TruthGPT
Interactive demo following deep learning best practices
"""

import gradio as gr
import torch
import logging
from typing import List, Tuple, Optional, Dict, Any
import json
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import base64
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Modern imports
from optimization_core.modules.optimizers import UnifiedOptimizer, OptimizationLevel
from optimization_core.training import Trainer, TrainingConfig


class TruthGPTGradioInterface:
    """
    Gradio interface for TruthGPT model interaction
    Following best practices for user-friendly interfaces
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.logger = self._setup_logging()
        
        # Load model and configuration
        self._load_model(model_path)
        
        # Initialize interface
        self.interface = self._create_interface()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for Gradio interface"""
        logger = logging.getLogger("TruthGPTGradioInterface")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_model(self, model_path: Optional[str]):
        """Load model and configuration"""
        try:
            model_name = "microsoft/DialoGPT-medium"
            if model_path:
                model_name = model_path
                
            self.logger.info(f"Loading model: {model_name}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
            except OSError:
                 # Fallback for demo if internet/model missing
                self.logger.warning("Model load failed, using tiny random GPT2.")
                config = AutoConfig.from_pretrained("gpt2")
                config.n_layer = 2
                self.model = AutoModelForCausalLM.from_config(config)
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token

            # Optimize
            self.optimizer = UnifiedOptimizer(level=OptimizationLevel.ADVANCED)
            result = self.optimizer.optimize(self.model)
            self.model = result.optimized_model
            
            self.logger.info("Model loaded and optimized successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            # Do not crash, just log
    
    def _create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""
        with gr.Blocks(
            title="TruthGPT Interactive Demo",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
            }
            """
        ) as interface:
            
            gr.Markdown("# 🤖 TruthGPT Interactive Demo (Unified Architecture)")
            gr.Markdown("Advanced LLM optimization and interaction interface")
            
            with gr.Tabs():
                # Chat Interface Tab
                with gr.Tab("💬 Chat Interface"):
                    self._create_chat_interface()
                
                # Model Training Tab
                with gr.Tab("🏋️ Model Training"):
                    self._create_training_interface()
            
            return interface
    
    def _create_chat_interface(self):
        """Create chat interface"""
        chatbot = gr.Chatbot(height=400)
        msg_input = gr.Textbox(label="Your Message")
        with gr.Row():
            send_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear")
            
        def chat_response(message, history):
            if not message.strip():
                return history, ""
            
            try:
                # Prepare inputs
                inputs = self.tokenizer(message + self.tokenizer.eos_token, return_tensors="pt")
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=100,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Naive cleanup: remove prompt if repeated
                if response.startswith(message):
                    response = response[len(message):].strip()
                    
                history.append((message, response))
                return history, ""
            except Exception as e:
                history.append((message, f"Error: {e}"))
                return history, ""

        send_btn.click(chat_response, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input])
        clear_btn.click(lambda: [], outputs=[chatbot])

    def _create_training_interface(self):
        """Create model training interface"""
        with gr.Row():
            with gr.Column():
                training_data = gr.Textbox(label="Training Data (one line per example)", lines=10)
                train_btn = gr.Button("Start Training", variant="primary")
                
            with gr.Column():
                progress_text = gr.Textbox(label="Status", lines=10)
        
        def start_training(data):
            texts = [l.strip() for l in data.split('\n') if l.strip()]
            if not texts:
                return "No data provided."
            
            try:
                config = TrainingConfig(num_epochs=1, batch_size=2, use_wandb=False)
                trainer = Trainer(self.model, self.tokenizer, config)
                
                train_loader, val_loader, _ = trainer.prepare_data(texts)
                history = trainer.train(train_loader, val_loader)
                
                trainer.cleanup()
                return f"Training complete!\nLoss: {history['train_loss'][-1]:.4f}"
            except Exception as e:
                return f"Training failed: {e}"
            
        train_btn.click(start_training, inputs=[training_data], outputs=[progress_text])

    def launch(self, share: bool = False, server_name: str = "127.0.0.1", server_port: int = 7860, **kwargs):
        self.interface.launch(share=share, server_name=server_name, server_port=server_port, **kwargs)

if __name__ == "__main__":
    interface = TruthGPTGradioInterface()
    interface.launch()

