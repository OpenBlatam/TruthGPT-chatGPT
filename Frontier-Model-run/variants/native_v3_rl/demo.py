#!/usr/bin/env python3
"""
Demo script for Native DeepSeek-V3 with Reinforcement Learning
Demonstrates the capabilities of the Native V3 RL model.
"""

import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import numpy as np
from typing import Dict, List, Any

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from model import NativeV3RLForCausalLM, NativeV3RLConfig


class NativeV3RLDemo:
    """Demo class for Native V3 RL model."""
    
    def __init__(self, model_size: str = "small"):
        """Initialize the demo with specified model size."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = model_size
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create model configuration based on size
        self.config = self._create_config(model_size)
        
        # Create model
        print(f"Creating {model_size} Native V3 RL model...")
        self.model = NativeV3RLForCausalLM(self.config)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize with random weights (in practice, you'd load trained weights)
        self._initialize_weights()
        
        print(f"Model created with {self._count_parameters():,} parameters")
        print(f"Device: {self.device}")
    
    def _create_config(self, model_size: str) -> NativeV3RLConfig:
        """Create model configuration based on size."""
        base_config = {
            "vocab_size": 50257,  # GPT-2 vocab size
            "max_position_embeddings": 2048,
            "use_reinforcement_learning": True,
            "reward_types": ["accuracy", "fluency", "helpfulness", "safety"],
            "use_multi_objective": True,
            "use_ppo": True,
            "use_value_heads": True,
            "use_curiosity": True,
            "layer_norm_eps": 1e-6,
            "initializer_range": 0.02,
        }
        
        if model_size == "small":
            base_config.update({
                "hidden_size": 512,
                "num_hidden_layers": 8,
                "num_attention_heads": 8,
                "intermediate_size": 2048,
                "moe_intermediate_size": 256,
                "num_routed_experts": 16,
                "num_shared_experts": 1,
                "num_activated_experts": 3,
                "kv_lora_rank": 128,
                "qk_nope_head_dim": 64,
                "qk_rope_head_dim": 32,
                "v_head_dim": 64,
            })
        elif model_size == "medium":
            base_config.update({
                "hidden_size": 1024,
                "num_hidden_layers": 16,
                "num_attention_heads": 16,
                "intermediate_size": 4096,
                "moe_intermediate_size": 512,
                "num_routed_experts": 32,
                "num_shared_experts": 2,
                "num_activated_experts": 4,
                "kv_lora_rank": 256,
                "qk_nope_head_dim": 64,
                "qk_rope_head_dim": 32,
                "v_head_dim": 64,
            })
        elif model_size == "large":
            base_config.update({
                "hidden_size": 2048,
                "num_hidden_layers": 27,
                "num_attention_heads": 16,
                "intermediate_size": 10944,
                "moe_intermediate_size": 1408,
                "num_routed_experts": 64,
                "num_shared_experts": 2,
                "num_activated_experts": 6,
                "kv_lora_rank": 512,
                "qk_nope_head_dim": 128,
                "qk_rope_head_dim": 64,
                "v_head_dim": 128,
            })
        else:
            raise ValueError(f"Unknown model size: {model_size}")
        
        return NativeV3RLConfig(**base_config)
    
    def _initialize_weights(self):
        """Initialize model weights (simplified)."""
        for module in self.model.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _count_parameters(self) -> int:
        """Count total model parameters."""
        return sum(p.numel() for p in self.model.parameters())
    
    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> Dict[str, Any]:
        """Generate text with RL-aware sampling."""
        
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        print(f"Generating text for prompt: '{prompt}'")
        print(f"Parameters: max_length={max_length}, temperature={temperature}, top_k={top_k}, top_p={top_p}")
        
        with torch.no_grad():
            # Generate with RL
            outputs = self.model.generate_with_rl(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample
            )
        
        # Decode generated text
        generated_ids = outputs['generated_ids']
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Extract RL information
        log_probs = outputs.get('log_probs')
        values_history = outputs.get('values_history', {})
        
        # Compute average values for each reward type
        avg_values = {}
        for reward_type, values in values_history.items():
            if values is not None:
                avg_values[reward_type] = values.mean().item()
        
        return {
            'generated_text': generated_text,
            'prompt': prompt,
            'full_text': generated_text,
            'new_text': generated_text[len(prompt):],
            'avg_log_prob': log_probs.mean().item() if log_probs is not None else None,
            'avg_values': avg_values,
            'num_tokens': generated_ids.shape[1],
        }
    
    def analyze_attention(self, text: str) -> Dict[str, Any]:
        """Analyze attention patterns in the model."""
        
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                output_attentions=True,
                return_dict=True
            )
        
        attentions = outputs.get('attentions', [])
        
        if not attentions:
            return {"error": "No attention weights available"}
        
        # Analyze attention patterns
        analysis = {
            'num_layers': len(attentions),
            'num_heads': attentions[0].shape[1] if attentions else 0,
            'sequence_length': attentions[0].shape[-1] if attentions else 0,
            'attention_entropy': [],
            'attention_max': [],
        }
        
        for layer_idx, attention in enumerate(attentions):
            # Compute entropy for each head
            attention_probs = F.softmax(attention, dim=-1)
            entropy = -(attention_probs * torch.log(attention_probs + 1e-8)).sum(dim=-1)
            
            analysis['attention_entropy'].append(entropy.mean().item())
            analysis['attention_max'].append(attention_probs.max().item())
        
        return analysis
    
    def demonstrate_rl_features(self, prompt: str) -> Dict[str, Any]:
        """Demonstrate RL-specific features."""
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate with different sampling strategies
        results = {}
        
        # Standard generation
        with torch.no_grad():
            standard_output = self.model.generate_with_rl(
                input_ids=input_ids,
                max_length=50,
                temperature=1.0,
                do_sample=True
            )
        
        results['standard'] = {
            'text': self.tokenizer.decode(standard_output['generated_ids'][0], skip_special_tokens=True),
            'avg_values': {k: v.mean().item() if v is not None else 0.0 
                          for k, v in standard_output.get('values_history', {}).items()}
        }
        
        # High temperature (more creative)
        with torch.no_grad():
            creative_output = self.model.generate_with_rl(
                input_ids=input_ids,
                max_length=50,
                temperature=1.5,
                do_sample=True
            )
        
        results['creative'] = {
            'text': self.tokenizer.decode(creative_output['generated_ids'][0], skip_special_tokens=True),
            'avg_values': {k: v.mean().item() if v is not None else 0.0 
                          for k, v in creative_output.get('values_history', {}).items()}
        }
        
        # Low temperature (more conservative)
        with torch.no_grad():
            conservative_output = self.model.generate_with_rl(
                input_ids=input_ids,
                max_length=50,
                temperature=0.3,
                do_sample=True
            )
        
        results['conservative'] = {
            'text': self.tokenizer.decode(conservative_output['generated_ids'][0], skip_special_tokens=True),
            'avg_values': {k: v.mean().item() if v is not None else 0.0 
                          for k, v in conservative_output.get('values_history', {}).items()}
        }
        
        return results
    
    def demonstrate_moe_routing(self, text: str) -> Dict[str, Any]:
        """Demonstrate MoE expert routing."""
        
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                return_dict=True
            )
        
        aux_losses = outputs.get('aux_losses', [])
        
        if not aux_losses:
            return {"error": "No MoE auxiliary losses available"}
        
        # Analyze expert usage
        analysis = {
            'num_moe_layers': len(aux_losses),
            'load_balancing_losses': [aux_loss.get('load_balancing_loss', 0.0) 
                                    for aux_loss in aux_losses if isinstance(aux_loss, dict)],
            'avg_load_balancing_loss': 0.0,
        }
        
        if analysis['load_balancing_losses']:
            analysis['avg_load_balancing_loss'] = sum(analysis['load_balancing_losses']) / len(analysis['load_balancing_losses'])
        
        return analysis


def run_demo():
    """Run the demo."""
    print("=" * 60)
    print("Native DeepSeek-V3 with Reinforcement Learning Demo")
    print("=" * 60)
    
    # Create demo instance
    demo = NativeV3RLDemo(model_size="small")  # Use small model for demo
    
    # Demo prompts
    prompts = [
        "The future of artificial intelligence is",
        "In a world where technology advances rapidly,",
        "The most important lesson I learned today was",
        "Climate change is a serious issue because",
    ]
    
    print("\n1. Text Generation Demo")
    print("-" * 30)
    
    for i, prompt in enumerate(prompts[:2]):  # Limit to 2 prompts for demo
        print(f"\nPrompt {i+1}: {prompt}")
        result = demo.generate_text(prompt, max_length=80)
        print(f"Generated: {result['new_text']}")
        print(f"Average values: {result['avg_values']}")
    
    print("\n2. RL Features Demo")
    print("-" * 30)
    
    rl_prompt = "The key to happiness is"
    print(f"\nPrompt: {rl_prompt}")
    rl_results = demo.demonstrate_rl_features(rl_prompt)
    
    for strategy, result in rl_results.items():
        print(f"\n{strategy.capitalize()} generation:")
        print(f"Text: {result['text'][len(rl_prompt):]}")
        print(f"Values: {result['avg_values']}")
    
    print("\n3. Attention Analysis Demo")
    print("-" * 30)
    
    attention_text = "Attention mechanisms are important in transformers"
    print(f"\nAnalyzing: {attention_text}")
    attention_analysis = demo.analyze_attention(attention_text)
    
    if 'error' not in attention_analysis:
        print(f"Layers: {attention_analysis['num_layers']}")
        print(f"Heads per layer: {attention_analysis['num_heads']}")
        print(f"Average attention entropy: {np.mean(attention_analysis['attention_entropy']):.4f}")
    else:
        print(f"Error: {attention_analysis['error']}")
    
    print("\n4. MoE Routing Demo")
    print("-" * 30)
    
    moe_text = "Mixture of experts allows for efficient scaling"
    print(f"\nAnalyzing MoE routing for: {moe_text}")
    moe_analysis = demo.demonstrate_moe_routing(moe_text)
    
    if 'error' not in moe_analysis:
        print(f"MoE layers: {moe_analysis['num_moe_layers']}")
        print(f"Average load balancing loss: {moe_analysis['avg_load_balancing_loss']:.6f}")
    else:
        print(f"Error: {moe_analysis['error']}")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()