#!/usr/bin/env python3
"""
Tests Detallados de Información de Modelos
==========================================

Proporciona información detallada sobre cada modelo:
- Arquitectura interna
- Parámetros configurables
- Uso de memoria
- Velocidad de inferencia
- Casos de uso específicos
"""

import torch
import torch.nn as nn
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from papers.research.paper_longrope import LongRoPEModule, LongRoPEConfig
from papers.research.paper_adagrope import AdaGroPEModule, AdaGroPEConfig
from papers.research.paper_cepe import CEPEModule, CEPEConfig
from papers.research.paper_longreward import LongRewardModule, LongRewardConfig


def analyze_model_architecture(model, model_name: str):
    """Analiza la arquitectura de un modelo."""
    
    print(f"\n{'='*70}")
    print(f"ARCHITECTURE ANALYSIS: {model_name}")
    print(f"{'='*70}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    print(f"   Non-trainable: {total_params - trainable_params:,}")
    
    print(f"\n🏗️  Architecture Components:")
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        print(f"   {name}: {module_params:,} params")
        if hasattr(module, '__class__'):
            print(f"      Type: {module.__class__.__name__}")
    
    # Analizar parámetros específicos
    print(f"\n⚙️  Key Parameters:")
    if hasattr(model, 'alpha') and hasattr(model, 'beta'):
        print(f"   α (alpha): {model.alpha.item():.4f}")
        print(f"   β (beta): {model.beta.item():.4f}")
    if hasattr(model, 'scaling_factors'):
        print(f"   Scaling factors: {len(model.scaling_factors)} values")
        print(f"      Min: {model.scaling_factors.min().item():.4f}")
        print(f"      Max: {model.scaling_factors.max().item():.4f}")
        print(f"      Mean: {model.scaling_factors.mean().item():.4f}")
    if hasattr(model, 'num_groups'):
        print(f"   Number of groups: {model.num_groups}")
    if hasattr(model, 'group_assignments'):
        print(f"   Group assignments: {len(model.group_assignments)} positions")
        unique_groups = torch.unique(model.group_assignments).tolist()
        print(f"   Unique groups: {len(unique_groups)}")
    if hasattr(model, 'chunk_encoder'):
        encoder_params = sum(p.numel() for p in model.chunk_encoder.parameters())
        print(f"   Chunk encoder: {encoder_params:,} params")
    if hasattr(model, 'reward_model'):
        reward_params = sum(p.numel() for p in model.reward_model.parameters()) if model.reward_model else 0
        print(f"   Reward model: {reward_params:,} params")


def test_memory_usage(model, model_name: str, context_lengths: list):
    """Test de uso de memoria en diferentes longitudes de contexto."""
    
    print(f"\n{'='*70}")
    print(f"MEMORY USAGE ANALYSIS: {model_name}")
    print(f"{'='*70}")
    
    print(f"\n{'Context':<12} {'Input MB':<12} {'Output MB':<12} {'Total MB':<12} {'Ratio':<12}")
    print("-" * 70)
    
    for ctx_len in context_lengths:
        try:
            hidden_dim = 768
            batch_size = 2
            
            # Input
            hidden_states = torch.randn(batch_size, ctx_len, hidden_dim)
            input_mb = hidden_states.numel() * 4 / (1024**2)
            
            # Forward
            model.eval()
            with torch.no_grad():
                output, _ = model(hidden_states)
            
            output_mb = output.numel() * 4 / (1024**2)
            total_mb = input_mb + output_mb
            ratio = output_mb / input_mb if input_mb > 0 else 0
            
            print(f"{ctx_len:<12} {input_mb:<12.2f} {output_mb:<12.2f} {total_mb:<12.2f} {ratio:<12.4f}")
            
        except Exception as e:
            print(f"{ctx_len:<12} ERROR: {str(e)}")


def test_inference_speed(model, model_name: str, context_lengths: list, num_runs: int = 5):
    """Test de velocidad de inferencia."""
    
    print(f"\n{'='*70}")
    print(f"INFERENCE SPEED ANALYSIS: {model_name}")
    print(f"{'='*70}")
    
    print(f"\n{'Context':<12} {'Time (ms)':<15} {'Tokens/sec':<15} {'Speedup':<12}")
    print("-" * 70)
    
    baseline_time = None
    
    for ctx_len in context_lengths:
        try:
            hidden_dim = 768
            batch_size = 2
            hidden_states = torch.randn(batch_size, ctx_len, hidden_dim)
            
            model.eval()
            
            # Warmup
            with torch.no_grad():
                _ = model(hidden_states)
            
            # Timing
            times = []
            for _ in range(num_runs):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start = time.time()
                
                with torch.no_grad():
                    _ = model(hidden_states)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                elapsed = (time.time() - start) * 1000
                times.append(elapsed)
            
            avg_time = sum(times) / len(times)
            tokens_per_sec = (ctx_len * batch_size) / (avg_time / 1000)
            
            if baseline_time is None:
                baseline_time = avg_time
                speedup = 1.0
            else:
                speedup = baseline_time / avg_time
            
            print(f"{ctx_len:<12} {avg_time:<15.2f} {tokens_per_sec:<15.0f} {speedup:<12.2f}x")
            
        except Exception as e:
            print(f"{ctx_len:<12} ERROR: {str(e)}")


def test_configuration_options(model_class, config_class, model_name: str):
    """Test de diferentes configuraciones."""
    
    print(f"\n{'='*70}")
    print(f"CONFIGURATION OPTIONS: {model_name}")
    print(f"{'='*70}")
    
    base_config = {
        "hidden_dim": 768,
        "base_context_length": 2048,
        "extended_context_length": 8192
    }
    
    # Configuraciones específicas por modelo
    if model_name == "LongRoPE":
        configs = [
            {**base_config, "use_non_uniform_scaling": True},
            {**base_config, "use_non_uniform_scaling": False},
            {**base_config, "rope_dim": 32},
            {**base_config, "rope_dim": 128},
        ]
    elif model_name == "AdaGroPE":
        configs = [
            {**base_config, "num_groups": 4},
            {**base_config, "num_groups": 8},
            {**base_config, "num_groups": 16},
        ]
    elif model_name == "CEPE":
        configs = [
            {**base_config, "chunk_size": 512},
            {**base_config, "chunk_size": 1024},
            {**base_config, "chunk_size": 2048},
            {**base_config, "encoder_hidden_dim": 128},
            {**base_config, "encoder_hidden_dim": 512},
        ]
    elif model_name == "LongReward":
        configs = [
            {**base_config, "reward_temperature": 0.5},
            {**base_config, "reward_temperature": 1.0},
            {**base_config, "reward_temperature": 2.0},
            {**base_config, "reward_model_dim": 128},
            {**base_config, "reward_model_dim": 512},
        ]
    else:
        configs = [base_config]
    
    print(f"\nTesting {len(configs)} configurations...")
    
    for i, config_dict in enumerate(configs):
        try:
            config = config_class(**config_dict)
            model = model_class(config)
            
            # Test básico
            hidden_states = torch.randn(2, 4096, 768)
            model.eval()
            with torch.no_grad():
                output, metadata = model(hidden_states)
            
            print(f"\n✅ Config {i+1}:")
            for key, value in config_dict.items():
                if key not in base_config or config_dict[key] != base_config[key]:
                    print(f"   {key}: {value}")
            print(f"   Output shape: {output.shape}")
            
        except Exception as e:
            print(f"\n❌ Config {i+1} failed: {str(e)}")


def main():
    """Ejecuta todos los análisis detallados."""
    
    models = [
        (LongRoPEModule, LongRoPEConfig, "LongRoPE"),
        (AdaGroPEModule, AdaGroPEConfig, "AdaGroPE"),
        (CEPEModule, CEPEConfig, "CEPE"),
        (LongRewardModule, LongRewardConfig, "LongReward"),
    ]
    
    context_lengths = [2048, 4096, 8192, 16384]
    
    for model_class, config_class, model_name in models:
        try:
            # Crear modelo
            config = config_class(
                hidden_dim=768,
                base_context_length=2048,
                extended_context_length=16384
            )
            model = model_class(config)
            
            # Análisis de arquitectura
            analyze_model_architecture(model, model_name)
            
            # Análisis de memoria
            test_memory_usage(model, model_name, context_lengths)
            
            # Análisis de velocidad
            test_inference_speed(model, model_name, context_lengths)
            
            # Opciones de configuración
            test_configuration_options(model_class, config_class, model_name)
            
        except Exception as e:
            print(f"\n❌ Error testing {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("✅ All detailed analyses completed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()


