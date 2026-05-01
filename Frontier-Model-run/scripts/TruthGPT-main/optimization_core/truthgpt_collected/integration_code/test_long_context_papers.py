#!/usr/bin/env python3
"""
Test Script para Top 10 Papers de Extensión de Context Window
==============================================================

Prueba todos los papers implementados para extensión de contexto.
"""

import torch
import sys
from pathlib import Path

# Agregar path
sys.path.insert(0, str(Path(__file__).parent))

from papers.research.paper_longrope import LongRoPEModule, LongRoPEConfig
from papers.research.paper_longrope2 import LongRoPE2Module, LongRoPE2Config
from papers.research.paper_cepe import CEPEModule, CEPEConfig
from papers.research.paper_adagrope import AdaGroPEModule, AdaGroPEConfig
from papers.research.paper_lift import LIFTModule, LIFTConfig
from papers.research.paper_semantic_compression import SemanticCompressionModule, SemanticCompressionConfig
from papers.research.paper_longreward import LongRewardModule, LongRewardConfig
from papers.research.paper_efficient_long_context import EfficientLongContextModule, EfficientLongContextConfig
from papers.research.paper_focusllm import FocusLLMModule, FocusLLMConfig
from papers.research.paper_longembed import LongEmbedModule, LongEmbedConfig


def test_paper(name: str, module_class, config_class, **config_kwargs):
    """Prueba un paper individual."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    try:
        # Crear config
        config = config_class(hidden_dim=768, **config_kwargs)
        
        # Crear módulo
        module = module_class(config)
        
        # Test con contexto corto
        print(f"\n  📝 Test 1: Contexto corto ({config.base_context_length} tokens)")
        hidden_states_short = torch.randn(2, config.base_context_length, config.hidden_dim)
        output_short, metadata_short = module(hidden_states_short)
        print(f"     ✅ Input: {hidden_states_short.shape}")
        print(f"     ✅ Output: {output_short.shape}")
        print(f"     ✅ Extended: {metadata_short.get('extended', False)}")
        
        # Test con contexto largo
        extended_len = min(config.extended_context_length, 4096)  # Limitar para test rápido
        print(f"\n  📝 Test 2: Contexto largo ({extended_len} tokens)")
        hidden_states_long = torch.randn(2, extended_len, config.hidden_dim)
        output_long, metadata_long = module(hidden_states_long)
        print(f"     ✅ Input: {hidden_states_long.shape}")
        print(f"     ✅ Output: {output_long.shape}")
        print(f"     ✅ Extended: {metadata_long.get('extended', False)}")
        
        # Métricas
        metrics = module.get_metrics()
        print(f"\n  📊 Métricas:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"     - {key}: {value}")
        
        print(f"\n  ✅ {name} PASSED")
        return True
        
    except Exception as e:
        print(f"\n  ❌ {name} FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Ejecuta todos los tests."""
    print("="*60)
    print("TEST SUITE: Top 10 Papers de Extensión de Context Window")
    print("="*60)
    
    # Lista de papers a probar
    papers = [
        ("LongRoPE", LongRoPEModule, LongRoPEConfig, {
            'base_context_length': 2048,
            'extended_context_length': 2048000
        }),
        ("LongRoPE2", LongRoPE2Module, LongRoPE2Config, {
            'base_context_length': 2048,
            'extended_context_length': 2048000,
            'use_evolutionary_search': False  # Más rápido para test
        }),
        ("CEPE", CEPEModule, CEPEConfig, {
            'base_context_length': 2048,
            'extended_context_length': 131072
        }),
        ("AdaGroPE", AdaGroPEModule, AdaGroPEConfig, {
            'base_context_length': 2048,
            'extended_context_length': 32768
        }),
        ("LIFT", LIFTModule, LIFTConfig, {
            'base_context_length': 2048,
            'extended_context_length': 32768
        }),
        ("Semantic Compression", SemanticCompressionModule, SemanticCompressionConfig, {
            'base_context_length': 2048,
            'extended_context_length': 16384
        }),
        ("LongReward", LongRewardModule, LongRewardConfig, {
            'base_context_length': 2048,
            'extended_context_length': 32768
        }),
        ("Efficient Long Context", EfficientLongContextModule, EfficientLongContextConfig, {
            'base_context_length': 2048,
            'extended_context_length': 16384
        }),
        ("FocusLLM", FocusLLMModule, FocusLLMConfig, {
            'base_context_length': 2048,
            'extended_context_length': 65536
        }),
        ("LongEmbed", LongEmbedModule, LongEmbedConfig, {
            'base_context_length': 512,
            'extended_context_length': 32768
        }),
    ]
    
    results = []
    for name, module_class, config_class, config_kwargs in papers:
        success = test_paper(name, module_class, config_class, **config_kwargs)
        results.append((name, success))
    
    # Resumen
    print(f"\n{'='*60}")
    print("RESUMEN DE TESTS")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} papers pasaron los tests")
    
    if passed == total:
        print("\n  🎉 ¡Todos los papers funcionan correctamente!")
    else:
        print(f"\n  ⚠️  {total - passed} papers fallaron")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)



