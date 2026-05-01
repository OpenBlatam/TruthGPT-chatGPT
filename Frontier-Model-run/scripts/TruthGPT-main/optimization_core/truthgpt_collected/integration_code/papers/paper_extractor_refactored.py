#!/usr/bin/env python3
"""
Paper Extractor Refactored - Versión Refactorizada
===================================================

Refactorización completa:
- Usa MetadataExtractor del core
- Elimina duplicación
- Mejor organización
- AST parsing mejorado
"""

import ast
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
import logging

import sys
from pathlib import Path

# Add core to path
core_dir = Path(__file__).parent / 'core'
if str(core_dir) not in sys.path:
    sys.path.insert(0, str(core_dir))

from metadata_extractor import MetadataExtractor, PaperMetadata
from paper_registry_refactored import PaperRegistryRefactored

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExtractedPaperInfo:
    """Información extraída completa de un paper."""
    # Metadata básica (usa PaperMetadata)
    metadata: PaperMetadata
    
    # Información adicional de AST
    config_fields: Dict[str, Any] = field(default_factory=dict)
    module_methods: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Código fuente
    forward_code: Optional[str] = None
    init_code: Optional[str] = None
    description: str = ""


class PaperExtractorRefactored:
    """
    Extractor refactorizado que usa MetadataExtractor.
    
    Mejoras:
    - Usa MetadataExtractor unificado
    - AST parsing mejorado
    - Extracción de código fuente
    """
    
    def __init__(self):
        self.extracted_papers: Dict[str, ExtractedPaperInfo] = {}
    
    def extract(self, paper_file: Path) -> ExtractedPaperInfo:
        """Extrae información completa de un paper."""
        logger.info(f"🔍 Extracting from {paper_file.name}")
        
        content = paper_file.read_text(encoding='utf-8')
        
        # Usar MetadataExtractor para metadata básica
        category = self._determine_category(paper_file)
        metadata = MetadataExtractor.extract_from_file(paper_file, category)
        
        if not metadata:
            raise ValueError(f"Failed to extract metadata from {paper_file}")
        
        info = ExtractedPaperInfo(metadata=metadata)
        
        # Extraer información adicional de AST
        try:
            tree = ast.parse(content)
            self._extract_from_ast(tree, info, content)
        except SyntaxError as e:
            logger.warning(f"Syntax error in {paper_file.name}: {e}")
        
        # Extraer descripción del docstring
        info.description = self._extract_description(content)
        
        self.extracted_papers[metadata.paper_id] = info
        return info
    
    def _determine_category(self, paper_file: Path) -> str:
        """Determina categoría del paper por su ubicación."""
        path_parts = paper_file.parts
        for part in path_parts:
            if part in PaperRegistryRefactored.CATEGORY_DIRS.values():
                return part
        return 'research'  # Default
    
    def _extract_from_ast(self, tree: ast.AST, info: ExtractedPaperInfo, content: str):
        """Extrae información usando AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Extraer campos de Config
                if 'Config' in node.name:
                    for item in node.body:
                        if isinstance(item, ast.AnnAssign) and item.target:
                            field_name = item.target.id if isinstance(item.target, ast.Name) else None
                            if field_name:
                                default_value = self._get_ast_value(item.value) if item.value else None
                                info.config_fields[field_name] = default_value
                
                # Extraer métodos de Module
                elif 'Module' in node.name:
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            info.module_methods.append(item.name)
                            
                            # Extraer código de forward
                            if item.name == 'forward':
                                info.forward_code = self._extract_code_snippet(content, item)
                            
                            # Extraer código de __init__
                            if item.name == '__init__':
                                info.init_code = self._extract_code_snippet(content, item)
            
            # Extraer imports
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    info.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    info.imports.append(f"{module}.{alias.name}")
    
    def _extract_code_snippet(self, content: str, node: ast.FunctionDef) -> str:
        """Extrae snippet de código de una función."""
        lines = content.split('\n')
        start_line = node.lineno - 1
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else node.lineno + 50
        return '\n'.join(lines[start_line:end_line])
    
    def _get_ast_value(self, node: ast.AST) -> Any:
        """Obtiene valor de un nodo AST."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.NameConstant):
            return node.value
        elif isinstance(node, ast.List):
            return [self._get_ast_value(item) for item in node.elts]
        elif isinstance(node, ast.Dict):
            return {
                self._get_ast_value(k): self._get_ast_value(v)
                for k, v in zip(node.keys, node.values)
            }
        return None
    
    def _extract_description(self, content: str) -> str:
        """Extrae descripción del docstring."""
        docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
        if docstring_match:
            docstring = docstring_match.group(1)
            lines = docstring.split('\n')
            description_lines = []
            for line in lines[1:]:
                line = line.strip()
                if line and not line.startswith('-') and not line.startswith('='):
                    description_lines.append(line)
                elif line.startswith('-'):
                    break
            return ' '.join(description_lines)
        return ""
    
    def extract_all(self, papers_dir: Path) -> Dict[str, ExtractedPaperInfo]:
        """Extrae información de todos los papers."""
        logger.info(f"🔍 Extracting all papers from {papers_dir}")
        
        for category, dir_name in PaperRegistryRefactored.CATEGORY_DIRS.items():
            category_path = papers_dir / dir_name
            if not category_path.exists():
                continue
            
            for paper_file in category_path.glob("paper_*.py"):
                try:
                    info = self.extract(paper_file)
                    logger.info(f"  ✅ {info.metadata.paper_id}: {info.metadata.paper_name}")
                except Exception as e:
                    logger.error(f"  ❌ Failed to extract {paper_file.name}: {e}")
        
        return self.extracted_papers
    
    def export_json(self, output_file: Path):
        """Exporta información a JSON."""
        data = {}
        for paper_id, info in self.extracted_papers.items():
            data[paper_id] = {
                'metadata': {
                    'paper_id': info.metadata.paper_id,
                    'paper_name': info.metadata.paper_name,
                    'category': info.metadata.category,
                    'speedup': info.metadata.speedup,
                    'accuracy_improvement': info.metadata.accuracy_improvement,
                    'benchmarks': info.metadata.benchmarks,
                    'key_techniques': info.metadata.key_techniques,
                },
                'config_fields': info.config_fields,
                'module_methods': info.module_methods,
                'imports': info.imports,
                'description': info.description
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Exported {len(data)} papers to {output_file}")


from .core.paper_registry_refactored import PaperRegistryRefactored


