#!/usr/bin/env python3
"""
Paper Loader Refactored - Versión Refactorizada
================================================

Refactorización completa:
- Usa core modules
- Elimina duplicación
- Mejor organización
- Optimizaciones
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import sys
from pathlib import Path

# Add core to path
core_dir = Path(__file__).parent / 'core'
if str(core_dir) not in sys.path:
    sys.path.insert(0, str(core_dir))

from paper_registry_refactored import get_registry, PaperRegistryRefactored, PaperModule
import threading
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaperLoaderRefactored:
    """
    Loader refactorizado y optimizado.
    
    Mejoras:
    - Usa registry refactorizado
    - Cache mejorado
    - Validación mejorada
    - Batch loading optimizado
    """
    
    def __init__(
        self,
        papers_base_dir: Optional[Path] = None,
        enable_cache: bool = True,
        max_cache_size: int = 200
    ):
        self.registry = get_registry(papers_base_dir)
        self.enable_cache = enable_cache
        self.max_cache_size = max_cache_size
        self.loaded_instances: Dict[str, Tuple[Any, Any]] = {}
        self.load_stats = {
            'total_loads': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'failed_loads': 0,
            'total_time': 0.0
        }
    
    @lru_cache(maxsize=256)
    def load_paper_module(
        self,
        paper_id: str,
        config_kwargs: Optional[Dict[str, Any]] = None,
        force_reload: bool = False
    ) -> Optional[Tuple[Any, Any]]:
        """
        Carga un paper y crea instancias.
        
        Args:
            paper_id: ID del paper
            config_kwargs: Argumentos para config
            force_reload: Si True, fuerza recarga
        
        Returns:
            Tuple (config_instance, module_instance) o None
        """
        # Generar cache key
        config_str = str(sorted((config_kwargs or {}).items()))
        cache_key = f"{paper_id}_{hash(config_str)}"
        
        # Verificar cache
        if not force_reload and cache_key in self.loaded_instances:
            self.load_stats['cache_hits'] += 1
            return self.loaded_instances[cache_key]
        
        self.load_stats['cache_misses'] += 1
        start_time = time.time()
        
        try:
            # Cargar paper module
            paper_module = self.registry.load_paper(paper_id, force_reload=force_reload)
            
            if not paper_module or not paper_module.loaded:
                logger.error(f"Failed to load paper {paper_id}")
                self.load_stats['failed_loads'] += 1
                return None
            
            # Crear instancias
            config_kwargs = config_kwargs or {}
            config_instance = paper_module.config_class(**config_kwargs)
            module_instance = paper_module.module_class(config_instance)
            
            load_time = time.time() - start_time
            result = (config_instance, module_instance)
            
            # Guardar en cache
            if self.enable_cache:
                self._add_to_cache(cache_key, result)
            
            self.load_stats['total_loads'] += 1
            self.load_stats['total_time'] += load_time
            
            logger.debug(f"✅ Loaded {paper_id} in {load_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error loading {paper_id}: {e}")
            self.load_stats['failed_loads'] += 1
            return None
    
    def _add_to_cache(self, cache_key: str, result: Tuple[Any, Any]):
        """Añade al cache con límite."""
        if len(self.loaded_instances) >= self.max_cache_size:
            # Remover el más antiguo (FIFO)
            oldest_key = next(iter(self.loaded_instances))
            del self.loaded_instances[oldest_key]
        
        self.loaded_instances[cache_key] = result
    
    def load_papers_batch(
        self,
        paper_ids: List[str],
        config_kwargs_map: Optional[Dict[str, Dict[str, Any]]] = None,
        max_workers: int = 4
    ) -> Dict[str, Optional[Tuple[Any, Any]]]:
        """Carga múltiples papers en paralelo."""
        logger.info(f"📦 Loading {len(paper_ids)} papers in batch...")
        
        config_kwargs_map = config_kwargs_map or {}
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.load_paper_module,
                    paper_id,
                    config_kwargs_map.get(paper_id)
                ): paper_id
                for paper_id in paper_ids
            }
            
            for future in as_completed(futures):
                paper_id = futures[future]
                try:
                    result = future.result()
                    results[paper_id] = result
                    if result:
                        logger.debug(f"  ✅ {paper_id}")
                    else:
                        logger.warning(f"  ⚠️  {paper_id} failed")
                except Exception as e:
                    logger.error(f"  ❌ {paper_id} error: {e}")
                    results[paper_id] = None
        
        logger.info(f"✅ Loaded {sum(1 for r in results.values() if r)}/{len(paper_ids)} papers")
        return results
    
    def validate_paper(self, paper_id: str) -> Tuple[bool, List[str]]:
        """Valida un paper."""
        errors = []
        
        if paper_id not in self.registry.registry:
            errors.append(f"Paper {paper_id} not found")
            return False, errors
        
        paper_module = self.registry.load_paper(paper_id)
        
        if not paper_module or not paper_module.loaded:
            errors.append(f"Failed to load: {paper_module.error if paper_module else 'Unknown'}")
            return False, errors
        
        if not paper_module.config_class:
            errors.append("Missing Config class")
        if not paper_module.module_class:
            errors.append("Missing Module class")
        
        # Validar instancias
        try:
            config = paper_module.config_class()
            module = paper_module.module_class(config)
            
            if not hasattr(module, 'forward'):
                errors.append("Module missing forward method")
            
            # Test forward
            test_input = torch.randn(2, 32, getattr(config, 'hidden_dim', 512))
            try:
                with torch.no_grad():
                    output = module(test_input)
                if output is None:
                    errors.append("Forward returns None")
            except Exception as e:
                errors.append(f"Forward error: {e}")
                
        except Exception as e:
            errors.append(f"Failed to create instances: {e}")
        
        return len(errors) == 0, errors
    
    def get_optimized_paper_list(
        self,
        requirements: Dict[str, Any],
        max_papers: Optional[int] = None
    ) -> List[str]:
        """Obtiene lista optimizada de papers."""
        candidates = []
        
        for paper_id, metadata in self.registry.registry.items():
            score = 0
            
            if 'category' in requirements:
                if metadata.category != requirements['category']:
                    continue
            
            if 'min_speedup' in requirements:
                if not metadata.speedup or metadata.speedup < requirements['min_speedup']:
                    continue
                score += metadata.speedup * 10
            
            if 'max_memory_impact' in requirements:
                impact_order = {'low': 3, 'medium': 2, 'high': 1}
                max_order = impact_order.get(requirements['max_memory_impact'], 2)
                if impact_order.get(metadata.memory_impact, 2) > max_order:
                    continue
                score += impact_order.get(metadata.memory_impact, 2)
            
            if 'min_accuracy' in requirements:
                if not metadata.accuracy_improvement or metadata.accuracy_improvement < requirements['min_accuracy']:
                    continue
                score += metadata.accuracy_improvement
            
            candidates.append((paper_id, score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        if max_papers:
            candidates = candidates[:max_papers]
        
        return [paper_id for paper_id, _ in candidates]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas."""
        registry_stats = self.registry.get_statistics()
        
        return {
            **self.load_stats,
            **registry_stats,
            'avg_load_time': (
                self.load_stats['total_time'] / self.load_stats['total_loads']
                if self.load_stats['total_loads'] > 0 else 0.0
            ),
            'cache_hit_rate': (
                self.load_stats['cache_hits'] / (self.load_stats['cache_hits'] + self.load_stats['cache_misses'])
                if (self.load_stats['cache_hits'] + self.load_stats['cache_misses']) > 0 else 0.0
            ),
            'loaded_instances': len(self.loaded_instances)
        }
    
    def clear_cache(self):
        """Limpia el cache."""
        self.loaded_instances.clear()
        self.load_paper_module.cache_clear()
        self.registry.clear_cache()
        logger.info("✅ Cache cleared")


# Singleton
_global_loader: Optional[PaperLoaderRefactored] = None
_loader_lock = threading.Lock()


def get_loader(papers_base_dir: Optional[Path] = None, **kwargs) -> PaperLoaderRefactored:
    """Obtiene el loader global."""
    global _global_loader
    if _global_loader is None:
        with _loader_lock:
            if _global_loader is None:
                _global_loader = PaperLoaderRefactored(papers_base_dir, **kwargs)
    return _global_loader


if __name__ == "__main__":
    # Test
    loader = PaperLoaderRefactored()
    
    print("\n" + "="*80)
    print("📚 PAPER LOADER REFACTORED TEST")
    print("="*80)
    
    papers = loader.registry.list_papers()
    print(f"\n✅ {len(papers)} papers available")
    
    if papers:
        first_paper = papers[0]
        print(f"\n🔍 Loading: {first_paper.paper_id}")
        
        config, module = loader.load_paper_module(
            first_paper.paper_id,
            {'hidden_dim': 512}
        )
        
        if config and module:
            print(f"  ✅ Loaded successfully")
            
            # Validar
            is_valid, errors = loader.validate_paper(first_paper.paper_id)
            print(f"  {'✅' if is_valid else '⚠️'} Validation: {is_valid}")
    
    stats = loader.get_statistics()
    print(f"\n📊 Statistics:")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"  Avg load time: {stats['avg_load_time']:.3f}s")


