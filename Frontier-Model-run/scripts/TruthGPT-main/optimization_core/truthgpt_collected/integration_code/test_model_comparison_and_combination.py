#!/usr/bin/env python3
"""
Tests Comprehensivos para Comparación y Combinación de Modelos de Contexto Largo
================================================================================

Este script proporciona:
1. Información detallada sobre cada modelo
2. Comparación de rendimiento y características
3. Tests de combinación de técnicas
4. Recomendaciones de uso según el caso
"""

import torch
import torch.nn as nn
import time
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import sys
import os

# Agregar path del proyecto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from papers.research.paper_longrope import LongRoPEModule, LongRoPEConfig
from papers.research.paper_adagrope import AdaGroPEModule, AdaGroPEConfig
from papers.research.paper_cepe import CEPEModule, CEPEConfig
from papers.research.paper_longreward import LongRewardModule, LongRewardConfig


@dataclass
class ModelInfo:
    """Información sobre un modelo."""
    name: str
    max_context: int
    base_context: int
    training_required: bool
    memory_efficient: bool
    speed: str  # "fast", "medium", "slow"
    best_for: List[str]
    limitations: List[str]
    params: Dict[str, Any]


@dataclass
class TestResult:
    """Resultado de un test."""
    model_name: str
    context_length: int
    batch_size: int
    hidden_dim: int
    forward_time_ms: float
    memory_mb: float
    output_shape: Tuple[int, ...]
    metadata: Dict[str, Any]
    success: bool
    error: str = ""


class ModelComparisonSuite:
    """Suite de comparación y combinación de modelos."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.model_infos: Dict[str, ModelInfo] = {}
        
    def get_model_info(self) -> Dict[str, ModelInfo]:
        """Obtiene información detallada sobre cada modelo."""
        
        self.model_infos = {
            "LongRoPE": ModelInfo(
                name="LongRoPE",
                max_context=2048000,  # 2M tokens
                base_context=2048,
                training_required=True,
                memory_efficient=True,
                speed="fast",
                best_for=[
                    "Contextos muy largos (hasta 2M tokens)",
                    "Aplicaciones que requieren fine-tuning",
                    "Cuando se necesita precisión en posiciones cercanas"
                ],
                limitations=[
                    "Requiere fine-tuning (1000 pasos)",
                    "Escalado no uniforme puede afectar posiciones lejanas"
                ],
                params={
                    "rope_dim": 64,
                    "scaling_factor": 1.0,
                    "use_non_uniform_scaling": True
                }
            ),
            "AdaGroPE": ModelInfo(
                name="AdaGroPE",
                max_context=32768,
                base_context=2048,
                training_required=False,  # Training-free!
                memory_efficient=True,
                speed="fast",
                best_for=[
                    "Plug-and-play sin entrenamiento",
                    "Extensión rápida de contexto",
                    "Cuando no se puede hacer fine-tuning"
                ],
                limitations=[
                    "Contexto máximo menor (32K vs 2M)",
                    "Granularidad gruesa en posiciones lejanas"
                ],
                params={
                    "num_groups": 8,
                    "base_context_length": 2048
                }
            ),
            "CEPE": ModelInfo(
                name="CEPE",
                max_context=131072,  # 128K
                base_context=2048,
                training_required=False,
                memory_efficient=True,  # Usa encoder pequeño
                speed="medium",
                best_for=[
                    "Contextos largos sin fine-tuning",
                    "Cuando se necesita compresión eficiente",
                    "Procesamiento paralelo de chunks"
                ],
                limitations=[
                    "Velocidad media (procesa chunks)",
                    "Requiere encoder adicional"
                ],
                params={
                    "chunk_size": 1024,
                    "encoder_hidden_dim": 256,
                    "compression_ratio": 0.5
                }
            ),
            "LongReward": ModelInfo(
                name="LongReward",
                max_context=32768,
                base_context=2048,
                training_required=True,
                memory_efficient=False,  # Requiere reward model
                speed="slow",
                best_for=[
                    "Optimización de dependencias largas",
                    "Cuando se necesita guía de atención",
                    "Mejora de calidad en contextos largos"
                ],
                limitations=[
                    "Requiere entrenamiento del reward model",
                    "Más lento (reward computation)",
                    "Mayor uso de memoria"
                ],
                params={
                    "reward_model_dim": 256,
                    "reward_temperature": 1.0,
                    "dependency_window": 512
                }
            )
        }
        
        return self.model_infos
    
    def test_model_performance(
        self,
        model_class,
        config_class,
        model_name: str,
        context_lengths: List[int] = [2048, 4096, 8192, 16384],
        batch_size: int = 2,
        hidden_dim: int = 768
    ) -> List[TestResult]:
        """Test de rendimiento de un modelo en diferentes longitudes de contexto."""
        
        print(f"\n{'='*60}")
        print(f"Testing {model_name}")
        print(f"{'='*60}")
        
        results = []
        
        # Configurar modelo con contexto máximo
        max_ctx = max(context_lengths)
        try:
            config = config_class(
                hidden_dim=hidden_dim,
                base_context_length=2048,
                extended_context_length=max_ctx
            )
            model = model_class(config)
            model.eval()
        except Exception as e:
            print(f"  ❌ Failed to initialize model: {str(e)}")
            return results
        
        for ctx_len in context_lengths:
            try:
                # Crear input
                hidden_states = torch.randn(batch_size, ctx_len, hidden_dim)
                
                # Medir tiempo
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                with torch.no_grad():
                    output, metadata = model(hidden_states)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                elapsed_ms = (time.time() - start_time) * 1000
                
                # Estimar memoria (aproximado)
                memory_mb = (hidden_states.numel() + output.numel()) * 4 / (1024**2)
                
                result = TestResult(
                    model_name=model_name,
                    context_length=ctx_len,
                    batch_size=batch_size,
                    hidden_dim=hidden_dim,
                    forward_time_ms=elapsed_ms,
                    memory_mb=memory_mb,
                    output_shape=tuple(output.shape),
                    metadata=metadata,
                    success=True
                )
                
                results.append(result)
                
                print(f"  ✅ Context {ctx_len:6d}: {elapsed_ms:6.2f}ms, {memory_mb:6.2f}MB")
                
            except Exception as e:
                result = TestResult(
                    model_name=model_name,
                    context_length=ctx_len,
                    batch_size=batch_size,
                    hidden_dim=hidden_dim,
                    forward_time_ms=0.0,
                    memory_mb=0.0,
                    output_shape=(),
                    metadata={},
                    success=False,
                    error=str(e)
                )
                results.append(result)
                print(f"  ❌ Context {ctx_len:6d}: ERROR - {str(e)[:60]}")
        
        self.results.extend(results)
        return results
    
    def test_model_combination(
        self,
        model1_class, config1_class, name1: str,
        model2_class, config2_class, name2: str,
        context_length: int = 8192,
        batch_size: int = 2,
        hidden_dim: int = 768
    ) -> Dict[str, Any]:
        """Test de combinación de dos modelos."""
        
        print(f"\n{'='*60}")
        print(f"Testing Combination: {name1} + {name2}")
        print(f"{'='*60}")
        
        try:
            # Crear modelos
            config1 = config1_class(
                hidden_dim=hidden_dim,
                base_context_length=2048,
                extended_context_length=context_length
            )
            model1 = model1_class(config1)
            model1.eval()
            
            config2 = config2_class(
                hidden_dim=hidden_dim,
                base_context_length=2048,
                extended_context_length=context_length
            )
            model2 = model2_class(config2)
            model2.eval()
            
            # Input
            hidden_states = torch.randn(batch_size, context_length, hidden_dim)
            
            # Combinación: Model1 -> Model2
            start_time = time.time()
            with torch.no_grad():
                output1, metadata1 = model1(hidden_states)
                output2, metadata2 = model2(output1)
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Verificar que funciona
            assert output2.shape == hidden_states.shape, "Shape mismatch!"
            
            print(f"  ✅ Combination successful!")
            print(f"     Time: {elapsed_ms:.2f}ms")
            print(f"     Output shape: {output2.shape}")
            
            return {
                "success": True,
                "time_ms": elapsed_ms,
                "output_shape": tuple(output2.shape),
                "metadata1": metadata1,
                "metadata2": metadata2
            }
            
        except Exception as e:
            print(f"  ❌ Combination failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_recommendations(self) -> Dict[str, Any]:
        """Genera recomendaciones basadas en los tests."""
        
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS")
        print(f"{'='*60}")
        
        recommendations = {
            "by_use_case": {
                "very_long_context_2M": {
                    "best": "LongRoPE",
                    "reason": "Único que soporta hasta 2M tokens",
                    "alternatives": ["CEPE (128K)"]
                },
                "no_training": {
                    "best": "AdaGroPE",
                    "reason": "Training-free, plug-and-play",
                    "alternatives": ["CEPE"]
                },
                "memory_constrained": {
                    "best": "AdaGroPE",
                    "reason": "Más eficiente en memoria",
                    "alternatives": ["CEPE", "LongRoPE"]
                },
                "quality_optimization": {
                    "best": "LongReward",
                    "reason": "Optimiza dependencias largas",
                    "alternatives": ["LongRoPE + LongReward"]
                },
                "fast_inference": {
                    "best": "AdaGroPE",
                    "reason": "Más rápido, sin overhead",
                    "alternatives": ["LongRoPE"]
                }
            },
            "combinations": {
                "AdaGroPE + LongReward": {
                    "description": "Extensión sin training + optimización de calidad",
                    "use_case": "Cuando necesitas contexto largo con alta calidad",
                    "tradeoff": "Más lento pero mejor calidad"
                },
                "LongRoPE + CEPE": {
                    "description": "Escalado RoPE + compresión paralela",
                    "use_case": "Contextos extremadamente largos con eficiencia",
                    "tradeoff": "Complejidad pero máxima extensión"
                },
                "CEPE + LongReward": {
                    "description": "Compresión + optimización",
                    "use_case": "Contextos largos con mejor manejo de dependencias",
                    "tradeoff": "Balance entre eficiencia y calidad"
                }
            },
            "performance_summary": {}
        }
        
        # Analizar resultados
        if self.results:
            by_model = {}
            for result in self.results:
                if result.success:
                    if result.model_name not in by_model:
                        by_model[result.model_name] = []
                    by_model[result.model_name].append(result)
            
            for model_name, results in by_model.items():
                avg_time = sum(r.forward_time_ms for r in results) / len(results)
                avg_memory = sum(r.memory_mb for r in results) / len(results)
                max_ctx = max(r.context_length for r in results)
                
                recommendations["performance_summary"][model_name] = {
                    "avg_time_ms": avg_time,
                    "avg_memory_mb": avg_memory,
                    "max_context_tested": max_ctx,
                    "num_tests": len(results)
                }
        
        return recommendations
    
    def print_comparison_table(self):
        """Imprime tabla comparativa de modelos."""
        
        print(f"\n{'='*80}")
        print("MODEL COMPARISON TABLE")
        print(f"{'='*80}")
        
        if not self.model_infos:
            self.get_model_info()
        
        print(f"\n{'Model':<15} {'Max Ctx':<12} {'Training':<12} {'Speed':<10} {'Memory':<12}")
        print("-" * 80)
        
        for name, info in self.model_infos.items():
            max_ctx_str = f"{info.max_context//1000}K" if info.max_context < 1000000 else f"{info.max_context//1000000}M"
            training_str = "Required" if info.training_required else "Free"
            print(f"{name:<15} {max_ctx_str:<12} {training_str:<12} {info.speed:<10} {'Efficient' if info.memory_efficient else 'Heavy':<12}")
        
        print(f"\n{'='*80}")
        print("BEST FOR:")
        print(f"{'='*80}")
        
        for name, info in self.model_infos.items():
            print(f"\n{name}:")
            for use_case in info.best_for:
                print(f"  • {use_case}")
        
        print(f"\n{'='*80}")
        print("LIMITATIONS:")
        print(f"{'='*80}")
        
        for name, info in self.model_infos.items():
            print(f"\n{name}:")
            for limitation in info.limitations:
                print(f"  ⚠ {limitation}")


def main():
    """Ejecuta todos los tests."""
    
    suite = ModelComparisonSuite()
    
    # Obtener información de modelos
    suite.get_model_info()
    suite.print_comparison_table()
    
    # Tests de rendimiento individual
    print(f"\n{'='*80}")
    print("PERFORMANCE TESTS")
    print(f"{'='*80}")
    
    suite.test_model_performance(
        LongRoPEModule, LongRoPEConfig, "LongRoPE",
        context_lengths=[2048, 4096, 8192, 16384]
    )
    
    suite.test_model_performance(
        AdaGroPEModule, AdaGroPEConfig, "AdaGroPE",
        context_lengths=[2048, 4096, 8192, 16384, 32768]
    )
    
    suite.test_model_performance(
        CEPEModule, CEPEConfig, "CEPE",
        context_lengths=[2048, 4096, 8192, 16384]
    )
    
    suite.test_model_performance(
        LongRewardModule, LongRewardConfig, "LongReward",
        context_lengths=[2048, 4096, 8192]
    )
    
    # Tests de combinación
    print(f"\n{'='*80}")
    print("COMBINATION TESTS")
    print(f"{'='*80}")
    
    suite.test_model_combination(
        AdaGroPEModule, AdaGroPEConfig, "AdaGroPE",
        LongRewardModule, LongRewardConfig, "LongReward",
        context_length=8192
    )
    
    suite.test_model_combination(
        LongRoPEModule, LongRoPEConfig, "LongRoPE",
        CEPEModule, CEPEConfig, "CEPE",
        context_length=8192
    )
    
    # Generar recomendaciones
    recommendations = suite.generate_recommendations()
    
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS BY USE CASE")
    print(f"{'='*80}")
    
    for use_case, rec in recommendations["by_use_case"].items():
        print(f"\n{use_case.replace('_', ' ').title()}:")
        print(f"  Best: {rec['best']}")
        print(f"  Reason: {rec['reason']}")
        print(f"  Alternatives: {', '.join(rec['alternatives'])}")
    
    print(f"\n{'='*80}")
    print("RECOMMENDED COMBINATIONS")
    print(f"{'='*80}")
    
    for combo, info in recommendations["combinations"].items():
        print(f"\n{combo}:")
        print(f"  Description: {info['description']}")
        print(f"  Use Case: {info['use_case']}")
        print(f"  Tradeoff: {info['tradeoff']}")
    
    # Guardar resultados
    output = {
        "model_infos": {k: asdict(v) for k, v in suite.model_infos.items()},
        "test_results": [asdict(r) for r in suite.results if r.success],
        "recommendations": recommendations
    }
    
    with open("model_comparison_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print("✅ All tests completed!")
    print("Results saved to: model_comparison_results.json")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()


