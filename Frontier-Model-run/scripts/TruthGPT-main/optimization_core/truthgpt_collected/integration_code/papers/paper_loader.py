#!/usr/bin/env python3
"""
Paper Loader - Sistema Optimizado de Carga de Papers
====================================================

Sistema rápido y eficiente para cargar papers usando:
- Registry system para descubrimiento automático
- Lazy loading (solo carga cuando se necesita)
- Cache inteligente
- Validación automática
- Batch loading para múltiples papers
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Type, Tuple, Set
from pathlib import Path
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from paper_registry import PaperRegistry, PaperModule, get_registry
from paper_extractor import PaperExtractor, ExtractedPaperInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaperLoader:
    """
    Loader optimizado para papers.
    
    Características:
    - Carga lazy (solo cuando se necesita)
    - Cache inteligente
    - Batch loading
    - Validación automática
    - Optimización de imports
    """
    
    def __init__(self, papers_base_dir: Optional[Path] = None, enable_cache: bool = True):
        self.registry = get_registry(papers_base_dir)
        self.extractor = PaperExtractor()
        self.enable_cache = enable_cache
        self.loaded_instances: Dict[str, Any] = {}  # Cache de instancias creadas
        self.load_stats = {
            'total_loads': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'failed_loads': 0,
            'total_time': 0.0
        }
    
    def load_paper_module(
        self,
        paper_id: str,
        config_kwargs: Optional[Dict[str, Any]] = None,
        force_reload: bool = False
    ) -> Optional[Tuple[Any, Any]]:
        """
        Carga un paper y crea instancias de Config y Module.
        
        Args:
            paper_id: ID del paper
            config_kwargs: Argumentos para la configuración
            force_reload: Si True, fuerza recarga
        
        Returns:
            Tuple (config_instance, module_instance) o None
        """
        cache_key = f"{paper_id}_{hash(str(config_kwargs))}"
        
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
                logger.error(f"Failed to load paper {paper_id}: {paper_module.error if paper_module else 'Unknown'}")
                self.load_stats['failed_loads'] += 1
                return None
            
            # Crear instancia de Config
            config_kwargs = config_kwargs or {}
            config_instance = paper_module.config_class(**config_kwargs)
            
            # Crear instancia de Module
            module_instance = paper_module.module_class(config_instance)
            
            load_time = time.time() - start_time
            
            result = (config_instance, module_instance)
            
            # Guardar en cache
            if self.enable_cache:
                self.loaded_instances[cache_key] = result
            
            self.load_stats['total_loads'] += 1
            self.load_stats['total_time'] += load_time
            
            logger.debug(f"✅ Loaded {paper_id} in {load_time:.3f}s")
            
            return result
            
        except Exception as e:
            load_time = time.time() - start_time
            logger.error(f"❌ Error loading {paper_id}: {e}")
            self.load_stats['failed_loads'] += 1
            return None
    
    def load_papers_batch(
        self,
        paper_ids: List[str],
        config_kwargs_map: Optional[Dict[str, Dict[str, Any]]] = None,
        max_workers: int = 4
    ) -> Dict[str, Optional[Tuple[Any, Any]]]:
        """
        Carga múltiples papers en paralelo.
        
        Args:
            paper_ids: Lista de IDs de papers
            config_kwargs_map: Mapa de paper_id -> config_kwargs
            max_workers: Número máximo de workers paralelos
        
        Returns:
            Dict de paper_id -> (config, module) o None
        """
        logger.info(f"📦 Loading {len(paper_ids)} papers in batch...")
        
        config_kwargs_map = config_kwargs_map or {}
        results = {}
        
        # Cargar en paralelo
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
    
    def load_papers_by_category(
        self,
        category: str,
        config_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Optional[Tuple[Any, Any]]]:
        """Carga todos los papers de una categoría."""
        papers = self.registry.list_papers(category=category)
        paper_ids = [p.paper_id for p in papers]
        return self.load_papers_batch(paper_ids, {pid: config_kwargs for pid in paper_ids})
    
    def get_paper_info(self, paper_id: str) -> Optional[ExtractedPaperInfo]:
        """Obtiene información extraída de un paper."""
        paper_metadata = self.registry.registry.get(paper_id)
        if not paper_metadata:
            return None
        
        # Extraer información completa
        return self.extractor.extract(paper_metadata.module_path)
    
    def validate_paper(self, paper_id: str) -> Tuple[bool, List[str]]:
        """
        Valida un paper antes de cargarlo.
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Verificar que existe en registry
        if paper_id not in self.registry.registry:
            errors.append(f"Paper {paper_id} not found in registry")
            return False, errors
        
        # Intentar cargar
        paper_module = self.registry.load_paper(paper_id)
        
        if not paper_module or not paper_module.loaded:
            errors.append(f"Failed to load: {paper_module.error if paper_module else 'Unknown'}")
            return False, errors
        
        # Verificar que tiene Config y Module
        if not paper_module.config_class:
            errors.append("Missing Config class")
        if not paper_module.module_class:
            errors.append("Missing Module class")
        
        # Intentar crear instancias
        try:
            config = paper_module.config_class()
            module = paper_module.module_class(config)
            
            # Verificar que tiene método forward
            if not hasattr(module, 'forward'):
                errors.append("Module missing forward method")
            
            # Verificar que forward funciona con input de prueba
            test_input = torch.randn(2, 32, config.hidden_dim if hasattr(config, 'hidden_dim') else 512)
            try:
                with torch.no_grad():
                    output = module(test_input)
                if output is None:
                    errors.append("Forward method returns None")
            except Exception as e:
                errors.append(f"Forward method error: {e}")
                
        except Exception as e:
            errors.append(f"Failed to create instances: {e}")
        
        return len(errors) == 0, errors
    
    def get_optimized_paper_list(
        self,
        requirements: Dict[str, Any],
        max_papers: Optional[int] = None
    ) -> List[str]:
        """
        Obtiene lista optimizada de papers basada en requisitos.
        
        Args:
            requirements: Dict con requisitos como:
                - 'category': categoría específica
                - 'min_speedup': speedup mínimo
                - 'max_memory_impact': impacto máximo en memoria
                - 'min_accuracy': mejora mínima de precisión
            max_papers: Número máximo de papers a retornar
        
        Returns:
            Lista de paper_ids optimizada
        """
        candidates = []
        
        for paper_id, metadata in self.registry.registry.items():
            score = 0
            
            # Filtrar por categoría
            if 'category' in requirements:
                if metadata.category != requirements['category']:
                    continue
            
            # Filtrar por speedup
            if 'min_speedup' in requirements:
                if not metadata.speedup or metadata.speedup < requirements['min_speedup']:
                    continue
                score += metadata.speedup * 10
            
            # Filtrar por memoria
            if 'max_memory_impact' in requirements:
                memory_scores = {'low': 3, 'medium': 2, 'high': 1}
                if memory_scores.get(metadata.memory_impact, 2) > memory_scores.get(requirements['max_memory_impact'], 2):
                    continue
                score += memory_scores.get(metadata.memory_impact, 2)
            
            # Filtrar por precisión
            if 'min_accuracy' in requirements:
                if not metadata.accuracy_improvement or metadata.accuracy_improvement < requirements['min_accuracy']:
                    continue
                score += metadata.accuracy_improvement
            
            candidates.append((paper_id, score))
        
        # Ordenar por score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Retornar top papers
        if max_papers:
            candidates = candidates[:max_papers]
        
        return [paper_id for paper_id, _ in candidates]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas del loader."""
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
        self.registry.clear_cache()
        logger.info("✅ Cache cleared")


# Singleton global
_global_loader: Optional[PaperLoader] = None


def get_loader(papers_base_dir: Optional[Path] = None) -> PaperLoader:
    """Obtiene el loader global (singleton)."""
    global _global_loader
    if _global_loader is None:
        _global_loader = PaperLoader(papers_base_dir)
    return _global_loader


def load_paper(paper_id: str, config_kwargs: Optional[Dict[str, Any]] = None) -> Optional[Tuple[Any, Any]]:
    """Carga un paper usando el loader global."""
    loader = get_loader()
    return loader.load_paper_module(paper_id, config_kwargs)


if __name__ == "__main__":
    # Test del loader
    papers_dir = Path(__file__).parent
    
    loader = PaperLoader(papers_dir)
    
    print("\n" + "="*80)
    print("📚 PAPER LOADER TEST")
    print("="*80)
    
    # Listar papers disponibles
    papers = loader.registry.list_papers()
    print(f"\n✅ {len(papers)} papers available")
    
    # Cargar un paper de ejemplo
    if papers:
        first_paper = papers[0]
        print(f"\n🔍 Loading: {first_paper.paper_id}")
        
        config, module = loader.load_paper_module(
            first_paper.paper_id,
            config_kwargs={'hidden_dim': 512}
        )
        
        if config and module:
            print(f"  ✅ Loaded successfully")
            print(f"  📦 Config: {type(config).__name__}")
            print(f"  📦 Module: {type(module).__name__}")
            
            # Validar
            is_valid, errors = loader.validate_paper(first_paper.paper_id)
            if is_valid:
                print(f"  ✅ Validation passed")
            else:
                print(f"  ⚠️  Validation errors: {errors}")
        else:
            print(f"  ❌ Failed to load")
    
    # Estadísticas
    stats = loader.get_statistics()
    print(f"\n📊 Statistics:")
    print(f"  Total loads: {stats['total_loads']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Cache misses: {stats['cache_misses']}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"  Avg load time: {stats['avg_load_time']:.3f}s")
    
    # Test de carga batch
    if len(papers) > 1:
        print(f"\n📦 Testing batch load...")
        paper_ids = [p.paper_id for p in papers[:3]]
        results = loader.load_papers_batch(paper_ids)
        print(f"  ✅ Loaded {sum(1 for r in results.values() if r)}/{len(paper_ids)} papers")



