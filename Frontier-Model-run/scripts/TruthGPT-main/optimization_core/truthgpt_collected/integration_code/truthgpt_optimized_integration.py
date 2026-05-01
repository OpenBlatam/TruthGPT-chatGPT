#!/usr/bin/env python3
"""
TruthGPT Optimized Integration - Sistema Optimizado con Registry
=================================================================

Integración optimizada que usa:
- Paper Registry para descubrimiento automático
- Paper Loader para carga eficiente
- Cache inteligente
- Validación automática
- Carga lazy de papers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
import time

# Importar sistema de papers optimizado
import sys
papers_dir = Path(__file__).parent / 'papers'
sys.path.insert(0, str(papers_dir))

from paper_registry import get_registry, PaperMetadata
from paper_loader import get_loader, PaperLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizedTruthGPTConfig:
    """Configuración optimizada para TruthGPT."""
    # Model dimensions
    vocab_size: int = 50257
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 1024
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    
    # Distance-based attention
    use_distance_attention: bool = True
    distance_type: str = "l1"
    lambda_param: float = 1.0
    use_learnable_lambda: bool = True
    
    # Papers to enable (usando IDs del registry)
    enabled_papers: List[str] = field(default_factory=list)
    
    # Paper-specific configs
    paper_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Performance optimizations
    enable_cache: bool = True
    lazy_load_papers: bool = True
    validate_papers: bool = True


class OptimizedTruthGPTModel(nn.Module):
    """
    Modelo TruthGPT optimizado con sistema de registry.
    
    Mejoras:
    - Carga automática de papers desde registry
    - Cache inteligente
    - Validación automática
    - Carga lazy
    """
    
    def __init__(self, config: OptimizedTruthGPTConfig):
        super().__init__()
        self.config = config
        self.loader = get_loader(papers_dir)
        self.registry = get_registry(papers_dir)
        
        # Componentes básicos
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Transformer blocks (simplificado para ejemplo)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.hidden_dropout_prob,
                batch_first=True
            )
            for _ in range(config.num_hidden_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Papers modules (cargados lazy)
        self.paper_modules: Dict[str, nn.Module] = {}
        self.paper_configs: Dict[str, Any] = {}
        self.paper_metadata: Dict[str, PaperMetadata] = {}
        
        # Cargar papers si no es lazy
        if not config.lazy_load_papers:
            self._load_papers()
        
        logger.info(f"✅ OptimizedTruthGPTModel initialized")
        logger.info(f"   Papers available: {len(self.registry.registry)}")
        logger.info(f"   Papers enabled: {len(config.enabled_papers)}")
    
    def _load_papers(self):
        """Carga los papers habilitados."""
        logger.info(f"📦 Loading {len(self.config.enabled_papers)} papers...")
        
        start_time = time.time()
        
        for paper_id in self.config.enabled_papers:
            try:
                # Validar si está habilitado
                if self.config.validate_papers:
                    is_valid, errors = self.loader.validate_paper(paper_id)
                    if not is_valid:
                        logger.warning(f"⚠️  Paper {paper_id} validation failed: {errors}")
                        continue
                
                # Obtener config específica del paper
                paper_config_kwargs = self.config.paper_configs.get(paper_id, {})
                paper_config_kwargs.setdefault('hidden_dim', self.config.hidden_size)
                
                # Cargar paper
                result = self.loader.load_paper_module(paper_id, paper_config_kwargs)
                
                if result:
                    paper_config, paper_module = result
                    self.paper_modules[paper_id] = paper_module
                    self.paper_configs[paper_id] = paper_config
                    
                    # Obtener metadata
                    if paper_id in self.registry.registry:
                        self.paper_metadata[paper_id] = self.registry.registry[paper_id]
                    
                    logger.info(f"  ✅ {paper_id}: {self.paper_metadata.get(paper_id, {}).paper_name or 'Unknown'}")
                else:
                    logger.warning(f"  ⚠️  Failed to load {paper_id}")
                    
            except Exception as e:
                logger.error(f"  ❌ Error loading {paper_id}: {e}")
        
        load_time = time.time() - start_time
        logger.info(f"✅ Loaded {len(self.paper_modules)}/{len(self.config.enabled_papers)} papers in {load_time:.3f}s")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_papers: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass optimizado.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: Optional [batch_size, seq_len]
            use_papers: Si aplicar papers habilitados
        
        Returns:
            Dict con logits y metadata
        """
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embeddings(position_ids)
        
        hidden_states = token_embeds + position_embeds
        hidden_states = self.embedding_dropout(hidden_states)
        
        # Transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, src_key_padding_mask=attention_mask)
        
        hidden_states = self.layer_norm(hidden_states)
        
        # Aplicar papers (carga lazy si es necesario)
        if use_papers:
            # Cargar papers lazy si no están cargados
            if self.config.lazy_load_papers:
                for paper_id in self.config.enabled_papers:
                    if paper_id not in self.paper_modules:
                        result = self.loader.load_paper_module(
                            paper_id,
                            self.config.paper_configs.get(paper_id, {'hidden_dim': self.config.hidden_size})
                        )
                        if result:
                            self.paper_modules[paper_id] = result[1]
                            self.paper_configs[paper_id] = result[0]
            
            # Aplicar papers secuencialmente
            paper_metadata_list = []
            for paper_id, paper_module in self.paper_modules.items():
                try:
                    # Aplicar paper
                    if hasattr(paper_module, 'forward'):
                        output = paper_module(hidden_states)
                        if isinstance(output, tuple):
                            hidden_states, metadata = output
                        else:
                            hidden_states = output
                            metadata = {}
                        
                        # Guardar metadata
                        if paper_id in self.paper_metadata:
                            paper_metadata_list.append({
                                'paper_id': paper_id,
                                'paper_name': self.paper_metadata[paper_id].paper_name,
                                'metadata': metadata
                            })
                except Exception as e:
                    logger.warning(f"⚠️  Error applying paper {paper_id}: {e}")
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        return {
            'logits': logits,
            'hidden_states': hidden_states,
            'paper_metadata': paper_metadata_list
        }
    
    def get_paper_info(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene información de un paper."""
        if paper_id in self.paper_metadata:
            metadata = self.paper_metadata[paper_id]
            return {
                'paper_id': metadata.paper_id,
                'paper_name': metadata.paper_name,
                'category': metadata.category,
                'speedup': metadata.speedup,
                'accuracy_improvement': metadata.accuracy_improvement,
                'benchmarks': metadata.benchmarks,
                'key_techniques': metadata.key_techniques,
                'loaded': paper_id in self.paper_modules
            }
        return None
    
    def list_enabled_papers(self) -> List[Dict[str, Any]]:
        """Lista información de todos los papers habilitados."""
        return [self.get_paper_info(pid) for pid in self.config.enabled_papers if self.get_paper_info(pid)]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas del modelo y papers."""
        loader_stats = self.loader.get_statistics()
        registry_stats = self.registry.get_statistics()
        
        return {
            'model': {
                'enabled_papers': len(self.config.enabled_papers),
                'loaded_papers': len(self.paper_modules),
                'total_parameters': sum(p.numel() for p in self.parameters()),
            },
            'loader': loader_stats,
            'registry': registry_stats
        }


class OptimizedTruthGPTCore:
    """
    Core optimizado de TruthGPT con sistema de registry.
    """
    
    def __init__(self, config: OptimizedTruthGPTConfig):
        self.config = config
        self.model = OptimizedTruthGPTModel(config)
        self.loader = get_loader(papers_dir)
        self.registry = get_registry(papers_dir)
    
    def enable_papers_by_category(self, category: str, config_kwargs: Optional[Dict[str, Any]] = None):
        """Habilita todos los papers de una categoría."""
        papers = self.registry.list_papers(category=category)
        paper_ids = [p.paper_id for p in papers]
        
        for paper_id in paper_ids:
            if paper_id not in self.config.enabled_papers:
                self.config.enabled_papers.append(paper_id)
                if config_kwargs:
                    self.config.paper_configs[paper_id] = config_kwargs
        
        logger.info(f"✅ Enabled {len(paper_ids)} papers from category '{category}'")
    
    def enable_papers_by_requirements(
        self,
        requirements: Dict[str, Any],
        max_papers: Optional[int] = None
    ):
        """Habilita papers basado en requisitos."""
        paper_ids = self.loader.get_optimized_paper_list(requirements, max_papers)
        
        for paper_id in paper_ids:
            if paper_id not in self.config.enabled_papers:
                self.config.enabled_papers.append(paper_id)
        
        logger.info(f"✅ Enabled {len(paper_ids)} papers based on requirements")
    
    def get_recommended_papers(self, use_case: str = "balanced") -> List[str]:
        """
        Obtiene papers recomendados para un caso de uso.
        
        Args:
            use_case: "speed", "accuracy", "efficiency", "balanced"
        """
        requirements_map = {
            "speed": {
                'min_speedup': 1.5,
                'max_memory_impact': 'medium'
            },
            "accuracy": {
                'min_accuracy': 10.0,
                'max_memory_impact': 'high'
            },
            "efficiency": {
                'max_memory_impact': 'low',
                'min_speedup': 1.2
            },
            "balanced": {
                'min_speedup': 1.0,
                'max_memory_impact': 'medium'
            }
        }
        
        requirements = requirements_map.get(use_case, requirements_map["balanced"])
        return self.loader.get_optimized_paper_list(requirements, max_papers=10)


if __name__ == "__main__":
    # Ejemplo de uso
    print("="*80)
    print("🚀 TRUTHGPT OPTIMIZED INTEGRATION TEST")
    print("="*80)
    
    # Crear configuración
    config = OptimizedTruthGPTConfig(
        hidden_size=768,
        num_hidden_layers=6,
        enabled_papers=[],  # Se pueden agregar después
        lazy_load_papers=True,
        validate_papers=True
    )
    
    # Crear core
    core = OptimizedTruthGPTCore(config)
    
    # Obtener papers recomendados
    print("\n📋 Recommended papers for 'balanced' use case:")
    recommended = core.get_recommended_papers("balanced")
    for paper_id in recommended[:5]:
        print(f"  - {paper_id}")
    
    # Habilitar algunos papers
    print("\n✅ Enabling recommended papers...")
    config.enabled_papers = recommended[:3]
    
    # Crear modelo
    model = OptimizedTruthGPTModel(config)
    
    # Test forward
    print("\n🧪 Testing forward pass...")
    input_ids = torch.randint(0, config.vocab_size, (2, 32))
    
    with torch.no_grad():
        output = model(input_ids, use_papers=True)
    
    print(f"  ✅ Output shape: {output['logits'].shape}")
    print(f"  📊 Papers applied: {len(output['paper_metadata'])}")
    
    # Estadísticas
    stats = model.get_statistics()
    print(f"\n📊 Statistics:")
    print(f"  Enabled papers: {stats['model']['enabled_papers']}")
    print(f"  Loaded papers: {stats['model']['loaded_papers']}")
    print(f"  Total parameters: {stats['model']['total_parameters']:,}")
    print(f"  Cache hit rate: {stats['loader']['cache_hit_rate']:.2%}")



