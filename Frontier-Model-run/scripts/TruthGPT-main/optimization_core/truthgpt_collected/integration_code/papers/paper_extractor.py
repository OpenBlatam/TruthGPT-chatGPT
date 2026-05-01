#!/usr/bin/env python3
"""
Paper Extractor - Sistema Mejorado de Extracción de Papers
===========================================================

Extrae información exacta de papers usando:
- Parsing avanzado de código Python
- Análisis de AST (Abstract Syntax Tree)
- Extracción de metadata estructurada
- Validación de papers
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExtractedPaperInfo:
    """Información extraída de un paper."""
    paper_id: str
    file_path: Path
    
    # Información básica
    title: str = ""
    description: str = ""
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    arxiv_id: Optional[str] = None
    url: Optional[str] = None
    
    # Clases y código
    config_class: Optional[str] = None
    module_class: Optional[str] = None
    config_fields: Dict[str, Any] = field(default_factory=dict)
    module_methods: List[str] = field(default_factory=list)
    
    # Técnicas y mejoras
    key_techniques: List[str] = field(default_factory=list)
    benchmarks: Dict[str, float] = field(default_factory=dict)
    improvements: Dict[str, Any] = field(default_factory=dict)
    
    # Performance
    speedup: Optional[float] = None
    accuracy_improvement: Optional[float] = None
    memory_impact: str = "medium"
    performance_impact: str = "medium"
    
    # Dependencias
    imports: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Código fuente
    forward_code: Optional[str] = None
    init_code: Optional[str] = None


class PaperExtractor:
    """Extractor avanzado de información de papers."""
    
    def __init__(self):
        self.extracted_papers: Dict[str, ExtractedPaperInfo] = {}
    
    def extract(self, paper_file: Path) -> ExtractedPaperInfo:
        """Extrae información completa de un paper."""
        logger.info(f"🔍 Extracting from {paper_file.name}")
        
        paper_id = paper_file.stem.replace('paper_', '').replace('_', '-')
        content = paper_file.read_text(encoding='utf-8')
        
        info = ExtractedPaperInfo(
            paper_id=paper_id,
            file_path=paper_file
        )
        
        # Parsear AST
        try:
            tree = ast.parse(content)
            self._extract_from_ast(tree, info, content)
        except SyntaxError as e:
            logger.warning(f"Syntax error in {paper_file.name}: {e}")
        
        # Extraer de docstring y comentarios
        self._extract_from_docstring(content, info)
        
        # Extraer de código (regex patterns)
        self._extract_from_patterns(content, info)
        
        self.extracted_papers[paper_id] = info
        return info
    
    def _extract_from_ast(self, tree: ast.AST, info: ExtractedPaperInfo, content: str):
        """Extrae información usando AST."""
        for node in ast.walk(tree):
            # Extraer clases
            if isinstance(node, ast.ClassDef):
                if 'Config' in node.name:
                    info.config_class = node.name
                    # Extraer campos de Config
                    for item in node.body:
                        if isinstance(item, ast.AnnAssign) and item.target:
                            field_name = item.target.id if isinstance(item.target, ast.Name) else None
                            if field_name:
                                # Intentar obtener valor por defecto
                                default_value = None
                                if isinstance(item.value, (ast.Constant, ast.Str, ast.Num)):
                                    default_value = self._get_ast_value(item.value)
                                elif isinstance(item.value, ast.NameConstant):
                                    default_value = item.value.value
                                info.config_fields[field_name] = default_value
                
                elif 'Module' in node.name:
                    info.module_class = node.name
                    # Extraer métodos
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            info.module_methods.append(item.name)
                    
                    # Extraer código de forward si existe
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == 'forward':
                            start_line = item.lineno - 1
                            end_line = item.end_lineno if hasattr(item, 'end_lineno') else item.lineno + 20
                            lines = content.split('\n')
                            info.forward_code = '\n'.join(lines[start_line:end_line])
                
                # Extraer código de __init__
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                        start_line = item.lineno - 1
                        end_line = item.end_lineno if hasattr(item, 'end_lineno') else item.lineno + 50
                        lines = content.split('\n')
                        info.init_code = '\n'.join(lines[start_line:end_line])
            
            # Extraer imports
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    info.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    info.imports.append(f"{module}.{alias.name}")
    
    def _extract_from_docstring(self, content: str, info: ExtractedPaperInfo):
        """Extrae información del docstring."""
        # Buscar docstring principal
        docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
        if docstring_match:
            docstring = docstring_match.group(1)
            lines = docstring.split('\n')
            
            # Extraer título (primera línea no vacía)
            for line in lines:
                line = line.strip()
                if line and not line.startswith('=') and not line.startswith('-'):
                    if not info.title:
                        info.title = line
                    break
            
            # Extraer descripción
            description_lines = []
            for line in lines[1:]:
                line = line.strip()
                if line and not line.startswith('-') and not line.startswith('='):
                    description_lines.append(line)
                elif line.startswith('-'):
                    break
            info.description = ' '.join(description_lines)
            
            # Extraer técnicas (líneas que empiezan con -)
            for line in lines:
                if line.strip().startswith('-'):
                    technique = line.strip()[1:].strip()
                    if technique and len(technique) > 5:
                        info.key_techniques.append(technique)
    
    def _extract_from_patterns(self, content: str, info: ExtractedPaperInfo):
        """Extrae información usando patrones regex."""
        content_lower = content.lower()
        
        # Extraer arXiv ID
        arxiv_match = re.search(r'arxiv[_\s]*id[:\s]*(\d{4}\.\d{4,5})', content, re.IGNORECASE)
        if arxiv_match:
            info.arxiv_id = arxiv_match.group(1)
            info.url = f"https://arxiv.org/abs/{info.arxiv_id}"
        
        # Extraer año
        year_match = re.search(r'\b(202[0-5])\b', content)
        if year_match:
            info.year = int(year_match.group(1))
        
        # Extraer autores
        authors_match = re.search(r'authors?[:\s]*\[(.*?)\]', content, re.IGNORECASE)
        if authors_match:
            authors_str = authors_match.group(1)
            info.authors = [a.strip().strip('"\'') for a in authors_str.split(',')]
        
        # Extraer benchmarks
        benchmark_patterns = [
            r'(\w+)[:\s]*(\d+\.?\d*)%',
            r'(\d+\.?\d*)%\s+en\s+(\w+)',
            r'(\w+)[:\s]*(\d+\.?\d*)',
        ]
        for pattern in benchmark_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    try:
                        if match[0].replace('.', '').isdigit():
                            value, name = match
                            info.benchmarks[name] = float(value)
                        else:
                            name, value = match
                            if value.replace('.', '').isdigit():
                                info.benchmarks[name] = float(value)
                    except ValueError:
                        pass
        
        # Extraer speedup
        speedup_match = re.search(r'(\d+\.?\d*)x\s*(?:speedup|faster)', content, re.IGNORECASE)
        if speedup_match:
            info.speedup = float(speedup_match.group(1))
        
        # Extraer mejora de precisión
        accuracy_match = re.search(r'\+?(\d+\.?\d*)%\s*(?:improvement|increase|mejora|accuracy)', content, re.IGNORECASE)
        if accuracy_match:
            info.accuracy_improvement = float(accuracy_match.group(1))
        
        # Determinar impactos
        if any(word in content_lower for word in ['speedup', 'faster', '2x', '3x']):
            info.performance_impact = "high"
        elif any(word in content_lower for word in ['slow', 'slower', 'overhead']):
            info.performance_impact = "low"
        
        if any(word in content_lower for word in ['memory efficient', 'low memory']):
            info.memory_impact = "low"
        elif any(word in content_lower for word in ['memory intensive', 'high memory']):
            info.memory_impact = "high"
    
    def _get_ast_value(self, node: ast.AST) -> Any:
        """Obtiene el valor de un nodo AST."""
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
    
    def extract_all(self, papers_dir: Path) -> Dict[str, ExtractedPaperInfo]:
        """Extrae información de todos los papers en un directorio."""
        logger.info(f"🔍 Extracting all papers from {papers_dir}")
        
        category_dirs = ['research', 'architecture', 'inference', 'memory', 'redundancy']
        
        for category in category_dirs:
            category_path = papers_dir / category
            if not category_path.exists():
                continue
            
            for paper_file in category_path.glob("paper_*.py"):
                try:
                    info = self.extract(paper_file)
                    logger.info(f"  ✅ {info.paper_id}: {info.title}")
                except Exception as e:
                    logger.error(f"  ❌ Failed to extract {paper_file.name}: {e}")
        
        return self.extracted_papers
    
    def export_json(self, output_file: Path):
        """Exporta información extraída a JSON."""
        data = {}
        for paper_id, info in self.extracted_papers.items():
            data[paper_id] = {
                'paper_id': info.paper_id,
                'title': info.title,
                'description': info.description,
                'authors': info.authors,
                'year': info.year,
                'arxiv_id': info.arxiv_id,
                'url': info.url,
                'config_class': info.config_class,
                'module_class': info.module_class,
                'config_fields': info.config_fields,
                'module_methods': info.module_methods,
                'key_techniques': info.key_techniques,
                'benchmarks': info.benchmarks,
                'improvements': info.improvements,
                'speedup': info.speedup,
                'accuracy_improvement': info.accuracy_improvement,
                'memory_impact': info.memory_impact,
                'performance_impact': info.performance_impact,
                'imports': info.imports,
                'dependencies': info.dependencies,
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Exported {len(data)} papers to {output_file}")


if __name__ == "__main__":
    # Test del extractor
    papers_dir = Path(__file__).parent
    
    extractor = PaperExtractor()
    papers_info = extractor.extract_all(papers_dir)
    
    print("\n" + "="*80)
    print("📚 PAPER EXTRACTION TEST")
    print("="*80)
    
    print(f"\n✅ Extracted {len(papers_info)} papers")
    
    # Mostrar algunos ejemplos
    for paper_id, info in list(papers_info.items())[:3]:
        print(f"\n📄 {paper_id}:")
        print(f"  Title: {info.title}")
        print(f"  Config: {info.config_class}")
        print(f"  Module: {info.module_class}")
        print(f"  Techniques: {len(info.key_techniques)}")
        print(f"  Benchmarks: {len(info.benchmarks)}")
        if info.speedup:
            print(f"  Speedup: {info.speedup}x")
    
    # Exportar JSON
    output_file = papers_dir / "extracted_papers.json"
    extractor.export_json(output_file)
    print(f"\n✅ Exported to {output_file}")



