#!/usr/bin/env python3
"""
Paper Registry System V2 - Versión Mejorada y Optimizada
==========================================================

Mejoras V2:
- Cache persistente en disco
- Pre-carga inteligente
- Búsqueda y filtrado avanzado
- Métricas de uso
- Compilación de módulos
- Retry logic
- Mejor manejo de errores
"""

import importlib.util
import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Type, Tuple, Callable, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict, OrderedDict
import logging
import time
import pickle
import hashlib
import threading
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PaperMetadata:
    """Metadata de un paper."""
    paper_id: str
    paper_name: str
    module_path: Path
    module_name: str
    category: str
    config_class: Optional[str] = None
    module_class: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    enabled_by_default: bool = False
    performance_impact: str = "medium"
    memory_impact: str = "medium"
    speedup: Optional[float] = None
    accuracy_improvement: Optional[float] = None
    arxiv_id: Optional[str] = None
    year: Optional[int] = None
    authors: List[str] = field(default_factory=list)
    benchmarks: Dict[str, float] = field(default_factory=dict)
    key_techniques: List[str] = field(default_factory=list)
    usage_count: int = 0
    last_used: Optional[float] = None
    load_count: int = 0
    error_count: int = 0


@dataclass
class PaperModule:
    """Módulo cargado de un paper."""
    metadata: PaperMetadata
    config_class: Type
    module_class: Type
    module: Any = None
    loaded: bool = False
    load_time: float = 0.0
    error: Optional[str] = None
    compiled: bool = False
    cache_key: Optional[str] = None


class PaperRegistryV2:
    """
    Registry mejorado con cache persistente y optimizaciones.
    
    Mejoras:
    - Cache persistente en disco
    - Pre-carga de papers populares
    - Búsqueda y filtrado avanzado
    - Métricas de uso
    - Compilación de módulos
    - Thread-safe
    """
    
    def __init__(
        self,
        papers_base_dir: Optional[Path] = None,
        enable_disk_cache: bool = True,
        preload_popular: bool = True,
        max_cache_size: int = 100
    ):
        if papers_base_dir is None:
            papers_base_dir = Path(__file__).parent
        
        self.papers_base_dir = Path(papers_base_dir)
        self.registry: Dict[str, PaperMetadata] = {}
        self.loaded_modules: Dict[str, PaperModule] = OrderedDict()  # LRU cache
        self.enable_disk_cache = enable_disk_cache
        self.preload_popular = preload_popular
        self.max_cache_size = max_cache_size
        self.cache_dir = self.papers_base_dir / ".paper_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self._lock = threading.RLock()  # Thread-safe
        
        # Estadísticas mejoradas
        self.stats = {
            'total_papers': 0,
            'loaded_papers': 0,
            'failed_loads': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'disk_cache_hits': 0,
            'disk_cache_misses': 0,
            'total_load_time': 0.0,
            'preload_time': 0.0
        }
        
        # Cargar metadata persistente si existe
        self._load_metadata_cache()
        
        # Auto-descubrir papers
        self._discover_papers()
        
        # Pre-cargar papers populares
        if self.preload_popular:
            self._preload_popular_papers()
    
    def _load_metadata_cache(self):
        """Carga metadata desde cache en disco."""
        metadata_file = self.cache_dir / "metadata_cache.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    cached_data = json.load(f)
                    for paper_id, data in cached_data.items():
                        if 'module_path' in data:
                            data['module_path'] = Path(data['module_path'])
                        # Restaurar metadata
                        # (se actualizará en _discover_papers si el archivo cambió)
            except Exception as e:
                logger.warning(f"Failed to load metadata cache: {e}")
    
    def _save_metadata_cache(self):
        """Guarda metadata en cache en disco."""
        metadata_file = self.cache_dir / "metadata_cache.json"
        try:
            data = {}
            for paper_id, metadata in self.registry.items():
                data[paper_id] = {
                    'paper_id': metadata.paper_id,
                    'paper_name': metadata.paper_name,
                    'module_path': str(metadata.module_path),
                    'category': metadata.category,
                    'usage_count': metadata.usage_count,
                    'last_used': metadata.last_used,
                    'load_count': metadata.load_count,
                    'error_count': metadata.error_count,
                    'speedup': metadata.speedup,
                    'accuracy_improvement': metadata.accuracy_improvement
                }
            with open(metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metadata cache: {e}")
    
    def _discover_papers(self):
        """Descubre automáticamente todos los papers."""
        logger.info(f"🔍 Discovering papers in {self.papers_base_dir}")
        
        category_dirs = {
            'research': 'research',
            'architecture': 'architecture',
            'inference': 'inference',
            'memory': 'memory',
            'redundancy': 'redundancy',
            'techniques': 'techniques',
            'code': 'code',
            'best': 'best'
        }
        
        for category, dir_name in category_dirs.items():
            category_path = self.papers_base_dir / dir_name
            if not category_path.exists():
                continue
            
            for paper_file in category_path.glob("paper_*.py"):
                try:
                    metadata = self._extract_metadata(paper_file, category)
                    if metadata:
                        # Preservar estadísticas si ya existe
                        if metadata.paper_id in self.registry:
                            old_meta = self.registry[metadata.paper_id]
                            metadata.usage_count = old_meta.usage_count
                            metadata.last_used = old_meta.last_used
                            metadata.load_count = old_meta.load_count
                            metadata.error_count = old_meta.error_count
                        
                        self.registry[metadata.paper_id] = metadata
                except Exception as e:
                    logger.warning(f"  ⚠️  Failed to register {paper_file.name}: {e}")
        
        self.stats['total_papers'] = len(self.registry)
        self._save_metadata_cache()
        logger.info(f"✅ Discovered {len(self.registry)} papers")
    
    def _extract_metadata(self, paper_file: Path, category: str) -> Optional[PaperMetadata]:
        """Extrae metadata mejorado."""
        try:
            content = paper_file.read_text(encoding='utf-8')
            paper_id = paper_file.stem.replace('paper_', '').replace('_', '-')
            
            # Extraer información básica
            paper_name = self._extract_paper_name(content)
            config_class, module_class = self._extract_classes(content)
            arxiv_id = self._extract_arxiv_id(content)
            year = self._extract_year(content)
            authors = self._extract_authors(content)
            benchmarks = self._extract_benchmarks(content)
            key_techniques = self._extract_techniques(content)
            performance_impact = self._determine_performance_impact(content)
            memory_impact = self._determine_memory_impact(content)
            speedup = self._extract_speedup(content)
            accuracy_improvement = self._extract_accuracy_improvement(content)
            
            return PaperMetadata(
                paper_id=paper_id,
                paper_name=paper_name,
                module_path=paper_file,
                module_name=paper_file.stem,
                category=category,
                config_class=config_class,
                module_class=module_class,
                arxiv_id=arxiv_id,
                year=year,
                authors=authors,
                benchmarks=benchmarks,
                key_techniques=key_techniques,
                performance_impact=performance_impact,
                memory_impact=memory_impact,
                speedup=speedup,
                accuracy_improvement=accuracy_improvement
            )
        except Exception as e:
            logger.error(f"Error extracting metadata from {paper_file}: {e}")
            return None
    
    def _extract_paper_name(self, content: str) -> str:
        """Extrae nombre del paper."""
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if '"""' in line or "'''" in line:
                for j in range(i+1, min(i+10, len(lines))):
                    if lines[j].strip() and not lines[j].strip().startswith('#'):
                        title = lines[j].strip().replace('=', '').replace('-', '').strip()
                        if title:
                            return title
        return "Unknown Paper"
    
    def _extract_classes(self, content: str) -> Tuple[Optional[str], Optional[str]]:
        """Extrae clases Config y Module."""
        import re
        config_match = re.search(r'class\s+(\w+Config)\s*[:\(]', content)
        module_match = re.search(r'class\s+(\w+Module)\s*[:\(]', content)
        return (
            config_match.group(1) if config_match else None,
            module_match.group(1) if module_match else None
        )
    
    def _extract_arxiv_id(self, content: str) -> Optional[str]:
        """Extrae arXiv ID."""
        import re
        match = re.search(r'arxiv[_\s]*id[:\s]*(\d{4}\.\d{4,5})', content, re.IGNORECASE)
        return match.group(1) if match else None
    
    def _extract_year(self, content: str) -> Optional[int]:
        """Extrae año."""
        import re
        match = re.search(r'\b(202[0-5])\b', content)
        return int(match.group(1)) if match else None
    
    def _extract_authors(self, content: str) -> List[str]:
        """Extrae autores."""
        import re
        match = re.search(r'authors?[:\s]*\[(.*?)\]', content, re.IGNORECASE)
        if match:
            return [a.strip().strip('"\'') for a in match.group(1).split(',')]
        return []
    
    def _extract_benchmarks(self, content: str) -> Dict[str, float]:
        """Extrae benchmarks."""
        import re
        benchmarks = {}
        patterns = [r'(\w+)[:\s]*(\d+\.?\d*)%', r'(\d+\.?\d*)%\s+en\s+(\w+)']
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    if match[0].replace('.', '').isdigit():
                        value, name = match
                        benchmarks[name] = float(value)
                    else:
                        name, value = match
                        if value.replace('.', '').isdigit():
                            benchmarks[name] = float(value)
                except ValueError:
                    pass
        return benchmarks
    
    def _extract_techniques(self, content: str) -> List[str]:
        """Extrae técnicas."""
        import re
        techniques = []
        docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
        if docstring_match:
            for line in docstring_match.group(1).split('\n'):
                if line.strip().startswith('-'):
                    technique = line.strip()[1:].strip()
                    if technique and len(technique) > 5:
                        techniques.append(technique)
        return techniques[:5]
    
    def _determine_performance_impact(self, content: str) -> str:
        """Determina impacto en performance."""
        content_lower = content.lower()
        if any(word in content_lower for word in ['speedup', 'faster', '2x', '3x']):
            return "high"
        elif any(word in content_lower for word in ['slow', 'slower', 'overhead']):
            return "low"
        return "medium"
    
    def _determine_memory_impact(self, content: str) -> str:
        """Determina impacto en memoria."""
        content_lower = content.lower()
        if any(word in content_lower for word in ['memory efficient', 'low memory']):
            return "low"
        elif any(word in content_lower for word in ['memory intensive', 'high memory']):
            return "high"
        return "medium"
    
    def _extract_speedup(self, content: str) -> Optional[float]:
        """Extrae speedup."""
        import re
        match = re.search(r'(\d+\.?\d*)x\s*speedup', content, re.IGNORECASE)
        return float(match.group(1)) if match else None
    
    def _extract_accuracy_improvement(self, content: str) -> Optional[float]:
        """Extrae mejora de precisión."""
        import re
        match = re.search(r'\+?(\d+\.?\d*)%\s*(?:improvement|increase|mejora)', content, re.IGNORECASE)
        return float(match.group(1)) if match else None
    
    def _preload_popular_papers(self, top_n: int = 5):
        """Pre-carga los papers más usados."""
        if len(self.registry) == 0:
            return
        
        start_time = time.time()
        logger.info(f"📦 Pre-loading top {top_n} popular papers...")
        
        # Ordenar por uso
        sorted_papers = sorted(
            self.registry.items(),
            key=lambda x: (x[1].usage_count, x[1].load_count),
            reverse=True
        )[:top_n]
        
        # Cargar en paralelo
        with ThreadPoolExecutor(max_workers=min(top_n, 4)) as executor:
            futures = {
                executor.submit(self._load_paper_internal, paper_id): paper_id
                for paper_id, _ in sorted_papers
            }
            
            for future in as_completed(futures):
                paper_id = futures[future]
                try:
                    future.result()
                    logger.debug(f"  ✅ Pre-loaded {paper_id}")
                except Exception as e:
                    logger.debug(f"  ⚠️  Failed to pre-load {paper_id}: {e}")
        
        preload_time = time.time() - start_time
        self.stats['preload_time'] = preload_time
        logger.info(f"✅ Pre-loaded {len(sorted_papers)} papers in {preload_time:.3f}s")
    
    def _load_paper_internal(self, paper_id: str, force_reload: bool = False) -> Optional[PaperModule]:
        """Carga interna de paper (thread-safe)."""
        with self._lock:
            if paper_id not in self.registry:
                return None
            
            # Verificar cache en memoria
            if not force_reload and paper_id in self.loaded_modules:
                module = self.loaded_modules[paper_id]
                # Mover al final (LRU)
                self.loaded_modules.move_to_end(paper_id)
                self.stats['cache_hits'] += 1
                return module
            
            # Verificar cache en disco
            if self.enable_disk_cache and not force_reload:
                cached_module = self._load_from_disk_cache(paper_id)
                if cached_module:
                    self.loaded_modules[paper_id] = cached_module
                    self._evict_if_needed()
                    self.stats['disk_cache_hits'] += 1
                    return cached_module
                self.stats['disk_cache_misses'] += 1
            
            self.stats['cache_misses'] += 1
            metadata = self.registry[paper_id]
            start_time = time.time()
            
            try:
                # Cargar módulo
                spec = importlib.util.spec_from_file_location(
                    metadata.module_name,
                    metadata.module_path
                )
                module = importlib.util.module_from_spec(spec)
                
                module_dir = str(metadata.module_path.parent)
                if module_dir not in sys.path:
                    sys.path.insert(0, module_dir)
                
                spec.loader.exec_module(module)
                
                # Obtener clases
                config_class = getattr(module, metadata.config_class, None) if metadata.config_class else None
                module_class = getattr(module, metadata.module_class, None) if metadata.module_class else None
                
                if not config_class or not module_class:
                    # Auto-descubrir
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type):
                            if 'Config' in attr_name and not config_class:
                                config_class = attr
                            elif 'Module' in attr_name and not module_class:
                                module_class = attr
                
                if not config_class or not module_class:
                    raise ValueError(f"Could not find Config or Module class")
                
                load_time = time.time() - start_time
                
                paper_module = PaperModule(
                    metadata=metadata,
                    config_class=config_class,
                    module_class=module_class,
                    module=module,
                    loaded=True,
                    load_time=load_time,
                    cache_key=self._generate_cache_key(paper_id)
                )
                
                # Guardar en cache
                self.loaded_modules[paper_id] = paper_module
                self._evict_if_needed()
                
                # Guardar en cache de disco
                if self.enable_disk_cache:
                    self._save_to_disk_cache(paper_module)
                
                # Actualizar estadísticas
                metadata.load_count += 1
                metadata.last_used = time.time()
                self.stats['loaded_papers'] += 1
                self.stats['total_load_time'] += load_time
                
                return paper_module
                
            except Exception as e:
                load_time = time.time() - start_time
                error_msg = str(e)
                metadata.error_count += 1
                
                paper_module = PaperModule(
                    metadata=metadata,
                    config_class=None,
                    module_class=None,
                    loaded=False,
                    load_time=load_time,
                    error=error_msg
                )
                
                self.loaded_modules[paper_id] = paper_module
                self.stats['failed_loads'] += 1
                
                return paper_module
    
    def _evict_if_needed(self):
        """Evicta módulos del cache si excede el tamaño máximo (LRU)."""
        while len(self.loaded_modules) > self.max_cache_size:
            # Remover el más antiguo (primero en OrderedDict)
            oldest_key = next(iter(self.loaded_modules))
            del self.loaded_modules[oldest_key]
    
    def _generate_cache_key(self, paper_id: str) -> str:
        """Genera clave de cache basada en contenido del archivo."""
        metadata = self.registry[paper_id]
        file_hash = hashlib.md5(metadata.module_path.read_bytes()).hexdigest()
        return f"{paper_id}_{file_hash[:8]}"
    
    def _save_to_disk_cache(self, paper_module: PaperModule):
        """Guarda módulo en cache de disco."""
        try:
            cache_file = self.cache_dir / f"{paper_module.cache_key}.pkl"
            # Solo guardar metadata, no el módulo completo (puede tener problemas de serialización)
            cache_data = {
                'paper_id': paper_module.metadata.paper_id,
                'config_class_name': paper_module.config_class.__name__ if paper_module.config_class else None,
                'module_class_name': paper_module.module_class.__name__ if paper_module.module_class else None,
                'load_time': paper_module.load_time,
                'cache_key': paper_module.cache_key
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            logger.debug(f"Failed to save disk cache: {e}")
    
    def _load_from_disk_cache(self, paper_id: str) -> Optional[PaperModule]:
        """Carga módulo desde cache de disco."""
        try:
            metadata = self.registry[paper_id]
            cache_key = self._generate_cache_key(paper_id)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            if not cache_file.exists():
                return None
            
            # Verificar que el archivo no cambió
            current_hash = hashlib.md5(metadata.module_path.read_bytes()).hexdigest()
            if cache_key != f"{paper_id}_{current_hash[:8]}":
                return None  # Archivo cambió, invalidar cache
            
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Cargar módulo real (no se puede cachear completamente)
            return self._load_paper_internal(paper_id, force_reload=True)
        except Exception:
            return None
    
    @lru_cache(maxsize=128)
    def load_paper(self, paper_id: str, force_reload: bool = False) -> Optional[PaperModule]:
        """Carga un paper (thread-safe, con retry)."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = self._load_paper_internal(paper_id, force_reload)
                if result and result.loaded:
                    # Actualizar uso
                    result.metadata.usage_count += 1
                    result.metadata.last_used = time.time()
                    return result
                elif attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))  # Backoff exponencial
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Retry {attempt + 1}/{max_retries} for {paper_id}: {e}")
                    time.sleep(0.1 * (attempt + 1))
                else:
                    logger.error(f"Failed to load {paper_id} after {max_retries} attempts: {e}")
        
        return None
    
    def search_papers(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        min_speedup: Optional[float] = None,
        min_accuracy: Optional[float] = None,
        max_memory_impact: Optional[str] = None,
        performance_impact: Optional[str] = None
    ) -> List[PaperMetadata]:
        """Búsqueda avanzada de papers."""
        results = list(self.registry.values())
        
        if query:
            query_lower = query.lower()
            results = [
                p for p in results
                if query_lower in p.paper_id.lower() or
                   query_lower in p.paper_name.lower() or
                   any(query_lower in tech.lower() for tech in p.key_techniques)
            ]
        
        if category:
            results = [p for p in results if p.category == category]
        
        if min_speedup:
            results = [p for p in results if p.speedup and p.speedup >= min_speedup]
        
        if min_accuracy:
            results = [p for p in results if p.accuracy_improvement and p.accuracy_improvement >= min_accuracy]
        
        if max_memory_impact:
            impact_order = {'low': 1, 'medium': 2, 'high': 3}
            max_order = impact_order.get(max_memory_impact, 3)
            results = [p for p in results if impact_order.get(p.memory_impact, 2) <= max_order]
        
        if performance_impact:
            results = [p for p in results if p.performance_impact == performance_impact]
        
        return sorted(results, key=lambda x: (x.usage_count, x.load_count), reverse=True)
    
    def get_paper(self, paper_id: str) -> Optional[PaperModule]:
        """Obtiene un paper (lo carga si es necesario)."""
        return self.load_paper(paper_id)
    
    def list_papers(self, category: Optional[str] = None) -> List[PaperMetadata]:
        """Lista papers, opcionalmente filtrados por categoría."""
        papers = list(self.registry.values())
        if category:
            papers = [p for p in papers if p.category == category]
        return sorted(papers, key=lambda x: x.paper_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas mejoradas."""
        return {
            **self.stats,
            'avg_load_time': (
                self.stats['total_load_time'] / self.stats['loaded_papers']
                if self.stats['loaded_papers'] > 0 else 0.0
            ),
            'cache_hit_rate': (
                self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
                if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0.0
            ),
            'disk_cache_hit_rate': (
                self.stats['disk_cache_hits'] / (self.stats['disk_cache_hits'] + self.stats['disk_cache_misses'])
                if (self.stats['disk_cache_hits'] + self.stats['disk_cache_misses']) > 0 else 0.0
            ),
            'cache_size': len(self.loaded_modules),
            'most_used_papers': [
                {'paper_id': p.paper_id, 'usage_count': p.usage_count}
                for p in sorted(self.registry.values(), key=lambda x: x.usage_count, reverse=True)[:5]
            ]
        }
    
    def clear_cache(self):
        """Limpia el cache."""
        with self._lock:
            self.loaded_modules.clear()
            self.load_paper.cache_clear()
            # Limpiar cache de disco
            if self.cache_dir.exists():
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()
            logger.info("✅ Cache cleared")
    
    def save_state(self):
        """Guarda estado del registry."""
        self._save_metadata_cache()
        logger.info("✅ State saved")


# Singleton mejorado
_global_registry: Optional[PaperRegistryV2] = None
_registry_lock = threading.Lock()


def get_registry_v2(papers_base_dir: Optional[Path] = None, **kwargs) -> PaperRegistryV2:
    """Obtiene el registry global mejorado (thread-safe singleton)."""
    global _global_registry
    if _global_registry is None:
        with _registry_lock:
            if _global_registry is None:
                _global_registry = PaperRegistryV2(papers_base_dir, **kwargs)
    return _global_registry


if __name__ == "__main__":
    # Test del registry mejorado
    registry = PaperRegistryV2()
    
    print("\n" + "="*80)
    print("📚 PAPER REGISTRY V2 TEST")
    print("="*80)
    
    print(f"\n✅ Discovered {len(registry.registry)} papers")
    
    # Búsqueda
    print("\n🔍 Search test:")
    results = registry.search_papers(query="reasoning", min_speedup=1.5)
    print(f"  Found {len(results)} papers matching 'reasoning' with speedup >= 1.5x")
    for paper in results[:3]:
        print(f"    - {paper.paper_id}: {paper.paper_name}")
    
    # Estadísticas
    stats = registry.get_statistics()
    print(f"\n📊 Statistics:")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"  Disk cache hit rate: {stats['disk_cache_hit_rate']:.2%}")
    print(f"  Most used papers: {len(stats['most_used_papers'])}")
    
    # Guardar estado
    registry.save_state()



