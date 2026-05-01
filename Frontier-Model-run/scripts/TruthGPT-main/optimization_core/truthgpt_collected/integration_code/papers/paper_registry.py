#!/usr/bin/env python3
"""
Paper Registry System - Sistema de Registro y Carga Rápida de Papers
=====================================================================

Sistema modular y eficiente para:
- Registrar papers automáticamente
- Cargar papers de forma lazy (solo cuando se necesitan)
- Cachear módulos cargados
- Validar papers antes de cargar
- Optimizar imports y carga
"""

import importlib.util
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Type, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import time
import pickle
import hashlib
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PaperMetadata:
    """Metadata de un paper."""
    paper_id: str
    paper_name: str
    module_path: Path
    module_name: str
    category: str  # research, architecture, inference, memory, etc.
    config_class: Optional[str] = None
    module_class: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    enabled_by_default: bool = False
    performance_impact: str = "medium"  # low, medium, high
    memory_impact: str = "medium"  # low, medium, high
    speedup: Optional[float] = None
    accuracy_improvement: Optional[float] = None
    arxiv_id: Optional[str] = None
    year: Optional[int] = None
    authors: List[str] = field(default_factory=list)
    benchmarks: Dict[str, float] = field(default_factory=dict)
    key_techniques: List[str] = field(default_factory=list)


@dataclass
class PaperModule:
    """Módulo cargado de un paper."""
    metadata: PaperMetadata
    config_class: Type
    module_class: Type
    module: Any = None  # El módulo importado
    loaded: bool = False
    load_time: float = 0.0
    error: Optional[str] = None


class PaperRegistry:
    """
    Registry centralizado para gestionar papers.
    
    Características:
    - Auto-descubrimiento de papers
    - Carga lazy (solo cuando se necesita)
    - Cache de módulos cargados
    - Validación automática
    - Optimización de imports
    """
    
    def __init__(self, papers_base_dir: Optional[Path] = None):
        if papers_base_dir is None:
            papers_base_dir = Path(__file__).parent
        
        self.papers_base_dir = Path(papers_base_dir)
        self.registry: Dict[str, PaperMetadata] = {}
        self.loaded_modules: Dict[str, PaperModule] = {}
        self.cache_enabled = True
        self.cache_dir = self.papers_base_dir / ".paper_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Estadísticas
        self.stats = {
            'total_papers': 0,
            'loaded_papers': 0,
            'failed_loads': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_load_time': 0.0
        }
        
        # Auto-descubrir papers
        self._discover_papers()
    
    def _discover_papers(self):
        """Descubre automáticamente todos los papers en el directorio."""
        logger.info(f"🔍 Discovering papers in {self.papers_base_dir}")
        
        # Mapeo de categorías a directorios
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
            
            # Buscar archivos paper_*.py
            for paper_file in category_path.glob("paper_*.py"):
                try:
                    metadata = self._extract_metadata(paper_file, category)
                    if metadata:
                        self.registry[metadata.paper_id] = metadata
                        logger.debug(f"  ✅ Registered: {metadata.paper_id}")
                except Exception as e:
                    logger.warning(f"  ⚠️  Failed to register {paper_file.name}: {e}")
        
        self.stats['total_papers'] = len(self.registry)
        logger.info(f"✅ Discovered {len(self.registry)} papers")
    
    def _extract_metadata(self, paper_file: Path, category: str) -> Optional[PaperMetadata]:
        """Extrae metadata de un archivo de paper."""
        try:
            # Leer el archivo para extraer información
            content = paper_file.read_text(encoding='utf-8')
            
            # Extraer paper_id del nombre del archivo
            paper_id = paper_file.stem.replace('paper_', '').replace('_', '-')
            
            # Extraer nombre del paper del docstring
            paper_name = self._extract_paper_name(content)
            
            # Extraer clases
            config_class, module_class = self._extract_classes(content)
            
            # Extraer información adicional del docstring
            arxiv_id = self._extract_arxiv_id(content)
            year = self._extract_year(content)
            authors = self._extract_authors(content)
            benchmarks = self._extract_benchmarks(content)
            key_techniques = self._extract_techniques(content)
            
            # Determinar performance impact
            performance_impact = self._determine_performance_impact(content)
            memory_impact = self._determine_memory_impact(content)
            
            # Extraer speedup y accuracy si están disponibles
            speedup = self._extract_speedup(content)
            accuracy_improvement = self._extract_accuracy_improvement(content)
            
            metadata = PaperMetadata(
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
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {paper_file}: {e}")
            return None
    
    def _extract_paper_name(self, content: str) -> str:
        """Extrae el nombre del paper del docstring."""
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if '"""' in line or "'''" in line:
                # Buscar título en las siguientes líneas
                for j in range(i+1, min(i+10, len(lines))):
                    if lines[j].strip() and not lines[j].strip().startswith('#'):
                        title = lines[j].strip()
                        # Limpiar título
                        title = title.replace('=', '').replace('-', '').strip()
                        if title:
                            return title
        return "Unknown Paper"
    
    def _extract_classes(self, content: str) -> Tuple[Optional[str], Optional[str]]:
        """Extrae nombres de clases Config y Module."""
        config_class = None
        module_class = None
        
        # Buscar patrones como "class XxxConfig" y "class XxxModule"
        import re
        config_match = re.search(r'class\s+(\w+Config)\s*[:\(]', content)
        module_match = re.search(r'class\s+(\w+Module)\s*[:\(]', content)
        
        if config_match:
            config_class = config_match.group(1)
        if module_match:
            module_class = module_match.group(1)
        
        return config_class, module_class
    
    def _extract_arxiv_id(self, content: str) -> Optional[str]:
        """Extrae arXiv ID del contenido."""
        import re
        match = re.search(r'arxiv[_\s]*id[:\s]*(\d{4}\.\d{4,5})', content, re.IGNORECASE)
        if match:
            return match.group(1)
        return None
    
    def _extract_year(self, content: str) -> Optional[int]:
        """Extrae año del paper."""
        import re
        # Buscar años 2020-2025
        match = re.search(r'\b(202[0-5])\b', content)
        if match:
            return int(match.group(1))
        return None
    
    def _extract_authors(self, content: str) -> List[str]:
        """Extrae autores del paper."""
        import re
        authors = []
        # Buscar patrones como "authors: [...]" o "Authors: ..."
        match = re.search(r'authors?[:\s]*\[(.*?)\]', content, re.IGNORECASE)
        if match:
            authors_str = match.group(1)
            authors = [a.strip().strip('"\'') for a in authors_str.split(',')]
        return authors
    
    def _extract_benchmarks(self, content: str) -> Dict[str, float]:
        """Extrae benchmarks y sus valores."""
        import re
        benchmarks = {}
        # Buscar patrones como "85.7% en AIME" o "AIME: 85.7%"
        patterns = [
            r'(\w+)[:\s]*(\d+\.?\d*)%',
            r'(\d+\.?\d*)%\s+en\s+(\w+)',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    if match[0].replace('.', '').isdigit():
                        value, name = match
                        benchmarks[name] = float(value)
                    else:
                        name, value = match
                        benchmarks[name] = float(value)
        return benchmarks
    
    def _extract_techniques(self, content: str) -> List[str]:
        """Extrae técnicas clave del paper."""
        import re
        techniques = []
        # Buscar en docstring o comentarios
        docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
        if docstring_match:
            docstring = docstring_match.group(1)
            # Buscar líneas que empiezan con "-"
            for line in docstring.split('\n'):
                if line.strip().startswith('-'):
                    technique = line.strip()[1:].strip()
                    if technique:
                        techniques.append(technique)
        return techniques[:5]  # Limitar a 5
    
    def _determine_performance_impact(self, content: str) -> str:
        """Determina el impacto en performance."""
        content_lower = content.lower()
        if any(word in content_lower for word in ['speedup', 'faster', '2x', '3x', 'fast']):
            return "high"
        elif any(word in content_lower for word in ['slow', 'slower', 'overhead']):
            return "low"
        return "medium"
    
    def _determine_memory_impact(self, content: str) -> str:
        """Determina el impacto en memoria."""
        content_lower = content.lower()
        if any(word in content_lower for word in ['memory efficient', 'low memory', 'efficient']):
            return "low"
        elif any(word in content_lower for word in ['memory intensive', 'high memory']):
            return "high"
        return "medium"
    
    def _extract_speedup(self, content: str) -> Optional[float]:
        """Extrae speedup si está disponible."""
        import re
        # Buscar patrones como "2.04x speedup" o "1.98x"
        match = re.search(r'(\d+\.?\d*)x\s*speedup', content, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None
    
    def _extract_accuracy_improvement(self, content: str) -> Optional[float]:
        """Extrae mejora de precisión si está disponible."""
        import re
        # Buscar patrones como "+25%" o "25% improvement"
        match = re.search(r'\+?(\d+\.?\d*)%\s*(?:improvement|increase|mejora)', content, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None
    
    @lru_cache(maxsize=128)
    def load_paper(self, paper_id: str, force_reload: bool = False) -> Optional[PaperModule]:
        """
        Carga un paper de forma lazy.
        
        Args:
            paper_id: ID del paper a cargar
            force_reload: Si True, fuerza recarga incluso si está en cache
        
        Returns:
            PaperModule cargado o None si falla
        """
        if paper_id not in self.registry:
            logger.error(f"Paper {paper_id} not found in registry")
            return None
        
        # Verificar cache
        if not force_reload and paper_id in self.loaded_modules:
            self.stats['cache_hits'] += 1
            return self.loaded_modules[paper_id]
        
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
            
            # Agregar al path si es necesario
            module_dir = str(metadata.module_path.parent)
            if module_dir not in sys.path:
                sys.path.insert(0, module_dir)
            
            # Ejecutar módulo
            spec.loader.exec_module(module)
            
            # Obtener clases
            config_class = None
            module_class = None
            
            if metadata.config_class:
                config_class = getattr(module, metadata.config_class, None)
            if metadata.module_class:
                module_class = getattr(module, metadata.module_class, None)
            
            if not config_class or not module_class:
                # Intentar auto-descubrir
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and 'Config' in attr_name:
                        config_class = attr
                    elif isinstance(attr, type) and 'Module' in attr_name:
                        module_class = attr
            
            if not config_class or not module_class:
                raise ValueError(f"Could not find Config or Module class in {paper_id}")
            
            load_time = time.time() - start_time
            
            paper_module = PaperModule(
                metadata=metadata,
                config_class=config_class,
                module_class=module_class,
                module=module,
                loaded=True,
                load_time=load_time
            )
            
            # Guardar en cache
            self.loaded_modules[paper_id] = paper_module
            self.stats['loaded_papers'] += 1
            self.stats['total_load_time'] += load_time
            
            logger.info(f"✅ Loaded {paper_id} in {load_time:.3f}s")
            
            return paper_module
            
        except Exception as e:
            load_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"❌ Failed to load {paper_id}: {error_msg}")
            
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
    
    def get_paper(self, paper_id: str) -> Optional[PaperModule]:
        """Obtiene un paper (lo carga si es necesario)."""
        return self.load_paper(paper_id)
    
    def list_papers(self, category: Optional[str] = None) -> List[PaperMetadata]:
        """Lista todos los papers, opcionalmente filtrados por categoría."""
        papers = list(self.registry.values())
        if category:
            papers = [p for p in papers if p.category == category]
        return sorted(papers, key=lambda x: x.paper_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas del registry."""
        return {
            **self.stats,
            'avg_load_time': (
                self.stats['total_load_time'] / self.stats['loaded_papers']
                if self.stats['loaded_papers'] > 0 else 0.0
            ),
            'cache_hit_rate': (
                self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
                if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0.0
            )
        }
    
    def clear_cache(self):
        """Limpia el cache de módulos cargados."""
        self.loaded_modules.clear()
        self.load_paper.cache_clear()
        logger.info("Cache cleared")


# Singleton global
_global_registry: Optional[PaperRegistry] = None


def get_registry(papers_base_dir: Optional[Path] = None) -> PaperRegistry:
    """Obtiene el registry global (singleton)."""
    global _global_registry
    if _global_registry is None:
        _global_registry = PaperRegistry(papers_base_dir)
    return _global_registry


def register_paper(paper_id: str, metadata: PaperMetadata):
    """Registra un paper manualmente."""
    registry = get_registry()
    registry.registry[paper_id] = metadata


def load_paper(paper_id: str) -> Optional[PaperModule]:
    """Carga un paper usando el registry global."""
    registry = get_registry()
    return registry.load_paper(paper_id)


if __name__ == "__main__":
    # Test del registry
    registry = PaperRegistry()
    
    print("\n" + "="*80)
    print("📚 PAPER REGISTRY TEST")
    print("="*80)
    
    print(f"\n✅ Discovered {len(registry.registry)} papers")
    
    # Listar papers
    print("\n📋 Papers by category:")
    for category in ['research', 'architecture', 'inference']:
        papers = registry.list_papers(category=category)
        if papers:
            print(f"\n  {category.upper()}:")
            for paper in papers[:5]:  # Mostrar primeros 5
                print(f"    - {paper.paper_id}: {paper.paper_name}")
    
    # Cargar un paper de ejemplo
    if registry.registry:
        first_paper_id = list(registry.registry.keys())[0]
        print(f"\n🔍 Loading paper: {first_paper_id}")
        paper_module = registry.load_paper(first_paper_id)
        
        if paper_module and paper_module.loaded:
            print(f"  ✅ Loaded successfully in {paper_module.load_time:.3f}s")
            print(f"  📦 Config: {paper_module.config_class}")
            print(f"  📦 Module: {paper_module.module_class}")
        else:
            print(f"  ❌ Failed to load: {paper_module.error if paper_module else 'Unknown error'}")
    
    # Estadísticas
    stats = registry.get_statistics()
    print(f"\n📊 Statistics:")
    print(f"  Total papers: {stats['total_papers']}")
    print(f"  Loaded: {stats['loaded_papers']}")
    print(f"  Failed: {stats['failed_loads']}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"  Avg load time: {stats['avg_load_time']:.3f}s")



