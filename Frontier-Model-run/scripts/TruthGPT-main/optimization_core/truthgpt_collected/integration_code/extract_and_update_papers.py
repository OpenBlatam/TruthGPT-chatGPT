#!/usr/bin/env python3
"""
Extract and Update Papers - Script de Extracción y Actualización
==================================================================

Script para:
- Extraer información de todos los papers
- Actualizar registry
- Generar reportes
- Validar papers
"""

import json
from pathlib import Path
from typing import Dict, List
import logging
from datetime import datetime

# Importar sistemas
import sys
papers_dir = Path(__file__).parent / 'papers'
sys.path.insert(0, str(papers_dir))

from paper_registry import PaperRegistry
from paper_extractor import PaperExtractor
from paper_loader import PaperLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_all_papers(papers_dir: Path) -> Dict:
    """Extrae información de todos los papers."""
    logger.info("🔍 Extracting all papers...")
    
    extractor = PaperExtractor()
    papers_info = extractor.extract_all(papers_dir)
    
    # Exportar JSON
    output_file = papers_dir / "extracted_papers.json"
    extractor.export_json(output_file)
    
    logger.info(f"✅ Extracted {len(papers_info)} papers")
    logger.info(f"📄 Exported to {output_file}")
    
    return papers_info


def update_registry(papers_dir: Path):
    """Actualiza el registry con todos los papers."""
    logger.info("📚 Updating registry...")
    
    registry = PaperRegistry(papers_dir)
    
    logger.info(f"✅ Registry updated: {len(registry.registry)} papers")
    
    return registry


def validate_all_papers(papers_dir: Path) -> Dict:
    """Valida todos los papers."""
    logger.info("✅ Validating all papers...")
    
    loader = PaperLoader(papers_dir)
    registry = get_registry(papers_dir)
    
    validation_results = {
        'valid': [],
        'invalid': [],
        'errors': {}
    }
    
    for paper_id in registry.registry.keys():
        is_valid, errors = loader.validate_paper(paper_id)
        
        if is_valid:
            validation_results['valid'].append(paper_id)
        else:
            validation_results['invalid'].append(paper_id)
            validation_results['errors'][paper_id] = errors
    
    logger.info(f"✅ Validation complete:")
    logger.info(f"   Valid: {len(validation_results['valid'])}")
    logger.info(f"   Invalid: {len(validation_results['invalid'])}")
    
    return validation_results


def generate_report(papers_dir: Path, output_file: Path):
    """Genera reporte completo de papers."""
    logger.info("📊 Generating report...")
    
    registry = PaperRegistry(papers_dir)
    extractor = PaperExtractor()
    
    report = []
    report.append("="*80)
    report.append("📚 PAPER REGISTRY REPORT")
    report.append("="*80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Resumen
    report.append("## 📊 SUMMARY")
    report.append("")
    report.append(f"- Total papers: {len(registry.registry)}")
    report.append("")
    
    # Papers por categoría
    report.append("## 📁 PAPERS BY CATEGORY")
    report.append("")
    
    categories = {}
    for paper_id, metadata in registry.registry.items():
        category = metadata.category
        if category not in categories:
            categories[category] = []
        categories[category].append(metadata)
    
    for category, papers in sorted(categories.items()):
        report.append(f"### {category.upper()} ({len(papers)} papers)")
        report.append("")
        report.append("| Paper ID | Name | Speedup | Accuracy | Status |")
        report.append("|----------|------|---------|----------|--------|")
        
        for paper in sorted(papers, key=lambda x: x.paper_id):
            speedup = f"{paper.speedup}x" if paper.speedup else "-"
            accuracy = f"+{paper.accuracy_improvement}%" if paper.accuracy_improvement else "-"
            status = "✅" if paper_id in registry.loaded_modules and registry.loaded_modules[paper_id].loaded else "⏳"
            
            report.append(f"| {paper.paper_id} | {paper.paper_name[:40]} | {speedup} | {accuracy} | {status} |")
        
        report.append("")
    
    # Top papers por speedup
    report.append("## ⚡ TOP PAPERS BY SPEEDUP")
    report.append("")
    
    papers_with_speedup = [
        (pid, meta) for pid, meta in registry.registry.items()
        if meta.speedup
    ]
    papers_with_speedup.sort(key=lambda x: x[1].speedup, reverse=True)
    
    report.append("| Rank | Paper ID | Speedup | Name |")
    report.append("|------|----------|---------|------|")
    for i, (paper_id, metadata) in enumerate(papers_with_speedup[:10], 1):
        report.append(f"| {i} | {paper_id} | {metadata.speedup}x | {metadata.paper_name[:50]} |")
    report.append("")
    
    # Top papers por accuracy
    report.append("## 🎯 TOP PAPERS BY ACCURACY")
    report.append("")
    
    papers_with_accuracy = [
        (pid, meta) for pid, meta in registry.registry.items()
        if meta.accuracy_improvement
    ]
    papers_with_accuracy.sort(key=lambda x: x[1].accuracy_improvement, reverse=True)
    
    report.append("| Rank | Paper ID | Accuracy | Name |")
    report.append("|------|----------|----------|------|")
    for i, (paper_id, metadata) in enumerate(papers_with_accuracy[:10], 1):
        report.append(f"| {i} | {paper_id} | +{metadata.accuracy_improvement}% | {metadata.paper_name[:50]} |")
    report.append("")
    
    # Estadísticas
    stats = registry.get_statistics()
    report.append("## 📈 STATISTICS")
    report.append("")
    report.append(f"- Total papers: {stats['total_papers']}")
    report.append(f"- Loaded papers: {stats['loaded_papers']}")
    report.append(f"- Failed loads: {stats['failed_loads']}")
    report.append(f"- Cache hit rate: {stats['cache_hit_rate']:.2%}")
    report.append(f"- Avg load time: {stats['avg_load_time']:.3f}s")
    report.append("")
    
    report.append("="*80)
    
    # Guardar reporte
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    logger.info(f"✅ Report saved to {output_file}")


def main():
    """Función principal."""
    print("="*80)
    print("🚀 EXTRACT AND UPDATE PAPERS")
    print("="*80)
    
    papers_dir = Path(__file__).parent / 'papers'
    output_dir = Path(__file__).parent
    
    # 1. Extraer papers
    print("\n📋 Step 1: Extracting papers...")
    papers_info = extract_all_papers(papers_dir)
    
    # 2. Actualizar registry
    print("\n📚 Step 2: Updating registry...")
    registry = update_registry(papers_dir)
    
    # 3. Validar papers
    print("\n✅ Step 3: Validating papers...")
    validation_results = validate_all_papers(papers_dir)
    
    # Guardar resultados de validación
    validation_file = output_dir / "paper_validation_results.json"
    with open(validation_file, 'w') as f:
        json.dump(validation_results, f, indent=2)
    logger.info(f"📄 Validation results saved to {validation_file}")
    
    # 4. Generar reporte
    print("\n📊 Step 4: Generating report...")
    report_file = output_dir / "PAPER_REGISTRY_REPORT.md"
    generate_report(papers_dir, report_file)
    
    # Resumen final
    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)
    print(f"📚 Total papers: {len(registry.registry)}")
    print(f"✅ Valid papers: {len(validation_results['valid'])}")
    print(f"❌ Invalid papers: {len(validation_results['invalid'])}")
    print(f"📄 Report: {report_file}")
    print(f"📄 Validation: {validation_file}")


if __name__ == "__main__":
    from paper_registry import get_registry
    main()



