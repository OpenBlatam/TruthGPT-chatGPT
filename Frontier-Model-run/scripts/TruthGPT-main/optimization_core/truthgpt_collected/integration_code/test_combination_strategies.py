#!/usr/bin/env python3
"""
Tests de Estrategias de Combinación de Modelos
==============================================

Explora diferentes formas de combinar modelos para obtener mejores resultados:
1. Combinaciones secuenciales
2. Combinaciones paralelas
3. Combinaciones adaptativas
4. Análisis de sinergias
"""

import torch
import torch.nn as nn
import time
import sys
import os
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from papers.research.paper_longrope import LongRoPEModule, LongRoPEConfig
from papers.research.paper_adagrope import AdaGroPEModule, AdaGroPEConfig
from papers.research.paper_cepe import CEPEModule, CEPEConfig
from papers.research.paper_longreward import LongRewardModule, LongRewardConfig


class SequentialCombination(nn.Module):
    """Combina modelos de forma secuencial."""
    
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, hidden_states: torch.Tensor, **kwargs):
        """Aplica modelos en secuencia."""
        x = hidden_states
        metadata_list = []
        
        for i, model in enumerate(self.models):
            x, metadata = model(x, **kwargs)
            metadata_list.append({f"model_{i}": metadata})
        
        return x, {"sequential": metadata_list}


class AdaptiveCombination(nn.Module):
    """Combina modelos adaptativamente según longitud de contexto."""
    
    def __init__(self, model1: nn.Module, model2: nn.Module, threshold: int = 4096):
        super().__init__()
        self.model1 = model1  # Para contextos cortos
        self.model2 = model2  # Para contextos largos
        self.threshold = threshold
    
    def forward(self, hidden_states: torch.Tensor, **kwargs):
        """Selecciona modelo según longitud."""
        seq_len = hidden_states.shape[1]
        
        if seq_len <= self.threshold:
            output, metadata = self.model1(hidden_states, **kwargs)
            metadata["selected_model"] = "model1"
        else:
            output, metadata = self.model2(hidden_states, **kwargs)
            metadata["selected_model"] = "model2"
        
        metadata["threshold"] = self.threshold
        metadata["context_length"] = seq_len
        
        return output, metadata


class ParallelCombination(nn.Module):
    """Combina modelos en paralelo y fusiona resultados."""
    
    def __init__(self, models: List[nn.Module], fusion_method: str = "mean"):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.fusion_method = fusion_method
    
    def forward(self, hidden_states: torch.Tensor, **kwargs):
        """Aplica modelos en paralelo y fusiona."""
        outputs = []
        metadata_list = []
        
        for model in self.models:
            output, metadata = model(hidden_states, **kwargs)
            outputs.append(output)
            metadata_list.append(metadata)
        
        # Fusionar outputs
        if self.fusion_method == "mean":
            fused = torch.stack(outputs).mean(dim=0)
        elif self.fusion_method == "weighted":
            # Pesos iguales por ahora
            weights = torch.ones(len(outputs)) / len(outputs)
            fused = sum(w * out for w, out in zip(weights, outputs))
        else:
            fused = outputs[0]  # Default: primer modelo
        
        return fused, {"parallel": metadata_list, "fusion": self.fusion_method}


def test_sequential_combinations():
    """Test de combinaciones secuenciales."""
    
    print(f"\n{'='*70}")
    print("SEQUENTIAL COMBINATIONS")
    print(f"{'='*70}")
    
    combinations = [
        ("AdaGroPE → LongReward", [
            (AdaGroPEModule, AdaGroPEConfig),
            (LongRewardModule, LongRewardConfig)
        ]),
        ("LongRoPE → CEPE", [
            (LongRoPEModule, LongRoPEConfig),
            (CEPEModule, CEPEConfig)
        ]),
        ("AdaGroPE → CEPE → LongReward", [
            (AdaGroPEModule, AdaGroPEConfig),
            (CEPEModule, CEPEConfig),
            (LongRewardModule, LongRewardConfig)
        ]),
    ]
    
    for combo_name, model_configs in combinations:
        print(f"\n📋 Testing: {combo_name}")
        
        try:
            models = []
            for model_class, config_class in model_configs:
                config = config_class(
                    hidden_dim=768,
                    base_context_length=2048,
                    extended_context_length=8192
                )
                models.append(model_class(config))
            
            combo = SequentialCombination(models)
            combo.eval()
            
            # Test
            hidden_states = torch.randn(2, 8192, 768)
            
            start_time = time.time()
            with torch.no_grad():
                output, metadata = combo(hidden_states)
            elapsed = (time.time() - start_time) * 1000
            
            print(f"   ✅ Success: {elapsed:.2f}ms")
            print(f"   Output shape: {output.shape}")
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")


def test_adaptive_combinations():
    """Test de combinaciones adaptativas."""
    
    print(f"\n{'='*70}")
    print("ADAPTIVE COMBINATIONS")
    print(f"{'='*70}")
    
    combinations = [
        ("AdaGroPE (short) + LongRoPE (long)", 
         AdaGroPEModule, AdaGroPEConfig,
         LongRoPEModule, LongRoPEConfig,
         4096),
        ("CEPE (short) + LongRoPE (long)",
         CEPEModule, CEPEConfig,
         LongRoPEModule, LongRoPEConfig,
         8192),
    ]
    
    context_lengths = [2048, 4096, 8192, 16384]
    
    for combo_name, m1_class, c1_class, m2_class, c2_class, threshold in combinations:
        print(f"\n📋 Testing: {combo_name} (threshold: {threshold})")
        
        try:
            # Modelo para contextos cortos
            config1 = c1_class(
                hidden_dim=768,
                base_context_length=2048,
                extended_context_length=threshold
            )
            model1 = m1_class(config1)
            
            # Modelo para contextos largos
            config2 = c2_class(
                hidden_dim=768,
                base_context_length=2048,
                extended_context_length=16384
            )
            model2 = m2_class(config2)
            
            combo = AdaptiveCombination(model1, model2, threshold)
            combo.eval()
            
            print(f"   {'Context':<12} {'Model':<20} {'Time (ms)':<12}")
            print("   " + "-" * 50)
            
            for ctx_len in context_lengths:
                hidden_states = torch.randn(2, ctx_len, 768)
                
                start_time = time.time()
                with torch.no_grad():
                    output, metadata = combo(hidden_states)
                elapsed = (time.time() - start_time) * 1000
                
                selected = metadata.get("selected_model", "unknown")
                print(f"   {ctx_len:<12} {selected:<20} {elapsed:<12.2f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")


def test_parallel_combinations():
    """Test de combinaciones paralelas."""
    
    print(f"\n{'='*70}")
    print("PARALLEL COMBINATIONS")
    print(f"{'='*70}")
    
    combinations = [
        ("AdaGroPE || LongRoPE", [
            (AdaGroPEModule, AdaGroPEConfig),
            (LongRoPEModule, LongRoPEConfig)
        ]),
        ("CEPE || AdaGroPE", [
            (CEPEModule, CEPEConfig),
            (AdaGroPEModule, AdaGroPEConfig)
        ]),
    ]
    
    for combo_name, model_configs in combinations:
        print(f"\n📋 Testing: {combo_name}")
        
        try:
            models = []
            for model_class, config_class in model_configs:
                config = config_class(
                    hidden_dim=768,
                    base_context_length=2048,
                    extended_context_length=8192
                )
                models.append(model_class(config))
            
            combo = ParallelCombination(models, fusion_method="mean")
            combo.eval()
            
            # Test
            hidden_states = torch.randn(2, 8192, 768)
            
            start_time = time.time()
            with torch.no_grad():
                output, metadata = combo(hidden_states)
            elapsed = (time.time() - start_time) * 1000
            
            print(f"   ✅ Success: {elapsed:.2f}ms")
            print(f"   Output shape: {output.shape}")
            print(f"   Fusion method: {metadata.get('fusion', 'unknown')}")
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")


def analyze_synergies():
    """Analiza sinergias entre modelos."""
    
    print(f"\n{'='*70}")
    print("SYNERGY ANALYSIS")
    print(f"{'='*70}")
    
    synergies = {
        "AdaGroPE + LongReward": {
            "description": "Extensión sin training + optimización de calidad",
            "benefits": [
                "AdaGroPE extiende contexto sin entrenamiento",
                "LongReward optimiza dependencias largas",
                "Mejor calidad en contextos extendidos"
            ],
            "tradeoffs": [
                "Más lento que usar solo AdaGroPE",
                "Requiere entrenamiento del reward model"
            ],
            "best_for": "Aplicaciones que necesitan contexto largo con alta calidad"
        },
        "LongRoPE + CEPE": {
            "description": "Escalado RoPE + compresión paralela",
            "benefits": [
                "LongRoPE extiende hasta 2M tokens",
                "CEPE comprime eficientemente",
                "Máxima extensión con eficiencia"
            ],
            "tradeoffs": [
                "Complejidad alta",
                "Requiere fine-tuning de LongRoPE"
            ],
            "best_for": "Contextos extremadamente largos (1M+ tokens)"
        },
        "CEPE + LongReward": {
            "description": "Compresión + optimización",
            "benefits": [
                "CEPE comprime eficientemente",
                "LongReward mejora calidad",
                "Balance entre eficiencia y calidad"
            ],
            "tradeoffs": [
                "Velocidad media",
                "Requiere entrenamiento"
            ],
            "best_for": "Contextos largos (64K-128K) con mejor calidad"
        }
    }
    
    for combo, info in synergies.items():
        print(f"\n🔗 {combo}:")
        print(f"   Description: {info['description']}")
        print(f"   Benefits:")
        for benefit in info['benefits']:
            print(f"      ✅ {benefit}")
        print(f"   Tradeoffs:")
        for tradeoff in info['tradeoffs']:
            print(f"      ⚠️  {tradeoff}")
        print(f"   Best for: {info['best_for']}")


def main():
    """Ejecuta todos los tests de combinación."""
    
    print(f"\n{'='*70}")
    print("MODEL COMBINATION STRATEGIES")
    print(f"{'='*70}")
    
    # Combinaciones secuenciales
    test_sequential_combinations()
    
    # Combinaciones adaptativas
    test_adaptive_combinations()
    
    # Combinaciones paralelas
    test_parallel_combinations()
    
    # Análisis de sinergias
    analyze_synergies()
    
    print(f"\n{'='*70}")
    print("✅ All combination tests completed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()


