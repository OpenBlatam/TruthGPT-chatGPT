#!/usr/bin/env python3
"""
Demo script for Frontier Model Variants
Showcases different model architectures and their capabilities.
"""

import os
import sys
import torch
import argparse
from typing import Dict, Any
import yaml

# Add variants to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'variants'))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import time

console = Console()


class FrontierModelDemo:
    """Demo class for frontier model variants."""
    
    def __init__(self):
        self.models = {}
        self.configs = {}
        
    def load_model(self, variant: str, config_path: str = None):
        """Load a specific model variant."""
        console.print(f"[yellow]Loading {variant} model...[/yellow]")
        
        # Load config
        if config_path is None:
            config_path = f"variants/{variant}/config.yaml"
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {"model_config": {}}
        
        # Create small demo config
        demo_config = {
            "vocab_size": 1000,  # Small vocab for demo
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "intermediate_size": 1024,
            "max_position_embeddings": 512,
        }
        
        # Update with variant-specific config
        demo_config.update(config.get('model_config', {}))
        demo_config.update({
            "vocab_size": 1000,  # Keep small for demo
            "hidden_size": 256,
            "num_hidden_layers": 4,
        })
        
        try:
            if variant == "native_transformer":
                from native_transformer.model import NativeTransformerForCausalLM, NativeTransformerConfig
                model_config = NativeTransformerConfig(**demo_config)
                model = NativeTransformerForCausalLM(model_config)
                
            elif variant == "mixture_of_experts":
                from mixture_of_experts.model import MoETransformerForCausalLM, MoETransformerConfig
                demo_config.update({
                    "num_experts": 4,  # Small number for demo
                    "num_experts_per_token": 2,
                })
                model_config = MoETransformerConfig(**demo_config)
                model = MoETransformerForCausalLM(model_config)
                
            elif variant == "retrieval_augmented":
                from retrieval_augmented.model import RAGTransformerForCausalLM, RAGTransformerConfig
                demo_config.update({
                    "retrieval_dim": 128,
                    "num_retrieved_docs": 3,
                })
                model_config = RAGTransformerConfig(**demo_config)
                model = RAGTransformerForCausalLM(model_config)
                
            elif variant == "multi_modal":
                from multi_modal.model import MultiModalTransformerForCausalLM, MultiModalTransformerConfig
                demo_config.update({
                    "modalities": ["text", "vision"],
                    "vision_hidden_size": 512,
                    "num_vision_tokens": 49,  # 7x7
                })
                model_config = MultiModalTransformerConfig(**demo_config)
                model = MultiModalTransformerForCausalLM(model_config)
                
            elif variant == "reinforcement_learning":
                from reinforcement_learning.model import RLTransformerForCausalLM, RLTransformerConfig
                demo_config.update({
                    "reward_types": ["accuracy", "fluency"],
                    "use_multi_objective": True,
                })
                model_config = RLTransformerConfig(**demo_config)
                model = RLTransformerForCausalLM(model_config)
                
            else:
                console.print(f"[red]Unknown variant: {variant}[/red]")
                return False
            
            self.models[variant] = model
            self.configs[variant] = model_config
            
            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())
            console.print(f"[green]✓ {variant} loaded ({num_params:,} parameters)[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]✗ Failed to load {variant}: {e}[/red]")
            return False

    def run_forward_pass(self, variant: str):
        """Run a forward pass through the model."""
        if variant not in self.models:
            console.print(f"[red]Model {variant} not loaded[/red]")
            return
        
        model = self.models[variant]
        model.eval()
        
        # Create dummy input
        batch_size = 2
        seq_length = 32
        vocab_size = self.configs[variant].vocab_size
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)
        
        console.print(f"[cyan]Running forward pass for {variant}...[/cyan]")
        
        try:
            with torch.no_grad():
                start_time = time.time()
                
                # Variant-specific forward pass
                if variant == "multi_modal":
                    # Add dummy vision input
                    images = torch.randn(batch_size, 3, 224, 224)
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        images=images
                    )
                elif variant == "reinforcement_learning":
                    # Add dummy rewards
                    rewards = {
                        "accuracy": torch.randn(batch_size, seq_length),
                        "fluency": torch.randn(batch_size, seq_length)
                    }
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        rewards=rewards,
                        compute_values=True
                    )
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                
                end_time = time.time()
                
                # Extract results
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
                
                console.print(f"[green]✓ Forward pass completed in {end_time - start_time:.3f}s[/green]")
                console.print(f"  Output shape: {logits.shape}")
                console.print(f"  Output range: [{logits.min():.3f}, {logits.max():.3f}]")
                
                # Variant-specific outputs
                if variant == "mixture_of_experts" and 'aux_losses' in outputs:
                    aux_losses = outputs['aux_losses']
                    console.print(f"  Auxiliary losses: {len(aux_losses)} layers")
                
                elif variant == "retrieval_augmented" and 'retrieval_info' in outputs:
                    retrieval_info = outputs['retrieval_info']
                    console.print(f"  Retrieval info: {len(retrieval_info)} layers")
                
                elif variant == "multi_modal" and 'cross_modal_info' in outputs:
                    cross_modal_info = outputs['cross_modal_info']
                    console.print(f"  Cross-modal info: {len(cross_modal_info)} layers")
                
                elif variant == "reinforcement_learning" and 'rl_info' in outputs:
                    rl_info = outputs['rl_info']
                    console.print(f"  RL info: {len(rl_info)} layers")
                    if 'value_loss' in outputs and outputs['value_loss'] is not None:
                        console.print(f"  Value loss: {outputs['value_loss']:.4f}")
                
        except Exception as e:
            console.print(f"[red]✗ Forward pass failed: {e}[/red]")

    def compare_models(self):
        """Compare loaded models."""
        if not self.models:
            console.print("[red]No models loaded[/red]")
            return
        
        table = Table(title="Model Comparison")
        table.add_column("Variant", style="cyan")
        table.add_column("Parameters", style="magenta")
        table.add_column("Hidden Size", style="green")
        table.add_column("Layers", style="yellow")
        table.add_column("Special Features", style="blue")
        
        for variant, model in self.models.items():
            config = self.configs[variant]
            num_params = sum(p.numel() for p in model.parameters())
            
            # Get special features
            features = []
            if variant == "native_transformer":
                features = ["Adaptive Attention", "RoPE", "RMS Norm"]
            elif variant == "mixture_of_experts":
                features = [f"{config.num_experts} Experts", "Load Balancing"]
            elif variant == "retrieval_augmented":
                features = ["Dense Retrieval", "Cross Attention"]
            elif variant == "multi_modal":
                features = ["Vision+Text", "Cross-Modal Fusion"]
            elif variant == "reinforcement_learning":
                features = ["Multi-Objective RL", "Value Heads"]
            
            table.add_row(
                variant,
                f"{num_params:,}",
                str(config.hidden_size),
                str(config.num_hidden_layers),
                ", ".join(features)
            )
        
        console.print(table)

    def benchmark_models(self):
        """Benchmark loaded models."""
        if not self.models:
            console.print("[red]No models loaded[/red]")
            return
        
        console.print("[cyan]Benchmarking models...[/cyan]")
        
        results = {}
        
        for variant in self.models:
            console.print(f"Benchmarking {variant}...")
            
            # Run multiple forward passes
            times = []
            for _ in range(5):
                start_time = time.time()
                self.run_forward_pass(variant)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
            results[variant] = avg_time
        
        # Display results
        table = Table(title="Benchmark Results")
        table.add_column("Variant", style="cyan")
        table.add_column("Avg Time (s)", style="magenta")
        table.add_column("Relative Speed", style="green")
        
        min_time = min(results.values())
        
        for variant, avg_time in sorted(results.items(), key=lambda x: x[1]):
            relative_speed = min_time / avg_time
            table.add_row(
                variant,
                f"{avg_time:.3f}",
                f"{relative_speed:.2f}x"
            )
        
        console.print(table)


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Demo Frontier Model Variants")
    parser.add_argument("--variants", nargs="+", 
                       choices=["native_transformer", "mixture_of_experts", 
                               "retrieval_augmented", "multi_modal", "reinforcement_learning"],
                       default=["native_transformer"],
                       help="Variants to demo")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmark comparison")
    parser.add_argument("--compare", action="store_true",
                       help="Compare model architectures")
    
    args = parser.parse_args()
    
    # Create demo instance
    demo = FrontierModelDemo()
    
    # Welcome message
    console.print(Panel.fit(
        "[bold blue]Frontier Model Variants Demo[/bold blue]\n"
        "Showcasing advanced transformer architectures",
        border_style="blue"
    ))
    
    # Load models
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        for variant in args.variants:
            task = progress.add_task(f"Loading {variant}...", total=None)
            success = demo.load_model(variant)
            progress.remove_task(task)
            
            if success:
                # Run forward pass demo
                demo.run_forward_pass(variant)
                console.print()
    
    # Compare models
    if args.compare and demo.models:
        console.print()
        demo.compare_models()
    
    # Benchmark models
    if args.benchmark and demo.models:
        console.print()
        demo.benchmark_models()
    
    console.print("\n[green]Demo completed![/green]")


if __name__ == "__main__":
    main()