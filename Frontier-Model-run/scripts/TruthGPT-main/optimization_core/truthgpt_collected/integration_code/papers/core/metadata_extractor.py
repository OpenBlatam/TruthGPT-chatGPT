#!/usr/bin/env python3
"""
Metadata Extractor - Clase Base para Extracción de Metadata
===========================================================

Extrae metadata de papers de forma unificada y eficiente.
Elimina duplicación entre registry y extractor.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class PaperMetadata:
    """Metadata unificada de un paper."""
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


class MetadataExtractor:
    """
    Extractor unificado de metadata.
    Elimina duplicación entre registry y extractor.
    """
    
    # Patrones regex compilados (mejor rendimiento)
    _ARXIV_PATTERN = re.compile(r'arxiv[_\s]*id[:\s]*(\d{4}[\._]\d{4,5})', re.IGNORECASE)
    _YEAR_PATTERN = re.compile(r'\b(202[0-5])\b')
    _AUTHORS_PATTERN = re.compile(r'authors?[:\s]*\[(.*?)\]', re.IGNORECASE)
    _SPEEDUP_PATTERN = re.compile(r'(\d+\.?\d*)x\s*speedup', re.IGNORECASE)
    _ACCURACY_PATTERN = re.compile(r'\+?(\d+\.?\d*)%\s*(?:improvement|increase|mejora)', re.IGNORECASE)
    _CONFIG_CLASS_PATTERN = re.compile(r'class\s+(\w+Config)\s*[:\(]')
    _MODULE_CLASS_PATTERN = re.compile(r'class\s+(\w+Module)\s*[:\(]')
    _BENCHMARK_PATTERNS = [
        re.compile(r'(\w+)[:\s]*(\d+\.?\d*)%', re.IGNORECASE),
        re.compile(r'(\d+\.?\d*)%\s+en\s+(\w+)', re.IGNORECASE),
    ]
    
    @classmethod
    def extract_from_file(cls, paper_file: Path, category: str) -> Optional[PaperMetadata]:
        """
        Extrae metadata de un archivo de paper.
        
        Args:
            paper_file: Path al archivo del paper
            category: Categoría del paper
        
        Returns:
            PaperMetadata o None si falla
        """
        try:
            content = paper_file.read_text(encoding='utf-8')
            paper_id = cls._extract_paper_id(paper_file)
            
            return PaperMetadata(
                paper_id=paper_id,
                paper_name=cls._extract_paper_name(content),
                module_path=paper_file,
                module_name=paper_file.stem,
                category=category,
                config_class=cls._extract_config_class(content),
                module_class=cls._extract_module_class(content),
                arxiv_id=cls._extract_arxiv_id(content, paper_file=paper_file),
                year=cls._extract_year(content),
                authors=cls._extract_authors(content),
                benchmarks=cls._extract_benchmarks(content),
                key_techniques=cls._extract_techniques(content),
                performance_impact=cls._determine_performance_impact(content),
                memory_impact=cls._determine_memory_impact(content),
                speedup=cls._extract_speedup(content),
                accuracy_improvement=cls._extract_accuracy_improvement(content)
            )
        except Exception as e:
            logger.error(f"Error extracting metadata from {paper_file}: {e}")
            return None
    
    @staticmethod
    def _extract_paper_id(paper_file: Path) -> str:
        """Extrae paper_id del nombre del archivo."""
        return paper_file.stem.replace('paper_', '').replace('_', '-')
    
    @classmethod
    def _extract_paper_name(cls, content: str) -> str:
        """Extrae nombre del paper del docstring."""
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if '"""' in line or "'''" in line:
                for j in range(i+1, min(i+10, len(lines))):
                    if lines[j].strip() and not lines[j].strip().startswith('#'):
                        title = lines[j].strip().replace('=', '').replace('-', '').strip()
                        if title:
                            return title
        return "Unknown Paper"
    
    @classmethod
    def _extract_config_class(cls, content: str) -> Optional[str]:
        """Extrae nombre de clase Config."""
        match = cls._CONFIG_CLASS_PATTERN.search(content)
        return match.group(1) if match else None
    
    @classmethod
    def _extract_module_class(cls, content: str) -> Optional[str]:
        """Extrae nombre de clase Module."""
        match = cls._MODULE_CLASS_PATTERN.search(content)
        return match.group(1) if match else None
    
    @classmethod
    def _extract_arxiv_id(cls, content: str, paper_file: Optional[Path] = None) -> Optional[str]:
        """Extrae arXiv ID del contenido o nombre de archivo."""
        # Intento 1: Por contenido
        match = cls._ARXIV_PATTERN.search(content)
        if match:
            return match.group(1).replace('_', '.')
            
        # Intento 2: Por nombre de archivo
        if paper_file:
            # Buscar patrón 1234.5678 o 1234_5678 en el nombre
            fn_match = re.search(r'(\d{4})[\._](\d{4,5})', paper_file.stem)
            if fn_match:
                return f"{fn_match.group(1)}.{fn_match.group(2)}"
                
        return None
    
    @classmethod
    def _extract_year(cls, content: str) -> Optional[int]:
        """Extrae año."""
        match = cls._YEAR_PATTERN.search(content)
        return int(match.group(1)) if match else None
    
    @classmethod
    def _extract_authors(cls, content: str) -> List[str]:
        """Extrae autores."""
        match = cls._AUTHORS_PATTERN.search(content)
        if match:
            return [a.strip().strip('"\'') for a in match.group(1).split(',')]
        return []
    
    @classmethod
    def _extract_benchmarks(cls, content: str) -> Dict[str, float]:
        """Extrae benchmarks."""
        benchmarks = {}
        for pattern in cls._BENCHMARK_PATTERNS:
            matches = pattern.findall(content)
            for match in matches:
                try:
                    if match[0].replace('.', '').isdigit():
                        value, name = match
                        benchmarks[name] = float(value)
                    else:
                        name, value = match
                        if value.replace('.', '').isdigit():
                            benchmarks[name] = float(value)
                except (ValueError, IndexError):
                    pass
        return benchmarks
    
    @classmethod
    def _extract_techniques(cls, content: str) -> List[str]:
        """Extrae técnicas clave."""
        techniques = []
        docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
        if docstring_match:
            for line in docstring_match.group(1).split('\n'):
                if line.strip().startswith('-'):
                    technique = line.strip()[1:].strip()
                    if technique and len(technique) > 5:
                        techniques.append(technique)
        return techniques[:5]
    
    @classmethod
    def _determine_performance_impact(cls, content: str) -> str:
        """Determina impacto en performance."""
        content_lower = content.lower()
        if any(word in content_lower for word in ['speedup', 'faster', '2x', '3x', 'fast']):
            return "high"
        elif any(word in content_lower for word in ['slow', 'slower', 'overhead']):
            return "low"
        return "medium"
    
    @classmethod
    def _determine_memory_impact(cls, content: str) -> str:
        """Determina impacto en memoria."""
        content_lower = content.lower()
        if any(word in content_lower for word in ['memory efficient', 'low memory', 'efficient']):
            return "low"
        elif any(word in content_lower for word in ['memory intensive', 'high memory']):
            return "high"
        return "medium"
    
    @classmethod
    def _extract_speedup(cls, content: str) -> Optional[float]:
        """Extrae speedup."""
        match = cls._SPEEDUP_PATTERN.search(content)
        return float(match.group(1)) if match else None
    
    @classmethod
    def _extract_accuracy_improvement(cls, content: str) -> Optional[float]:
        """Extrae mejora de precisión."""
        match = cls._ACCURACY_PATTERN.search(content)
        return float(match.group(1)) if match else None



