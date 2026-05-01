#!/usr/bin/env python3
"""
Script para ejecutar benchmarks del modelo Polyglot vs modelos closed source.

Este script facilita la ejecución de los benchmarks y genera reportes detallados.

Usage:
    python scripts/run_polyglot_benchmarks.py
    python scripts/run_polyglot_benchmarks.py --full --model gpt2
    python scripts/run_polyglot_benchmarks.py --modules kv_cache compression
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_polyglot_benchmark_vs_closed_source import (
    PolyglotBenchmarker,
    ClosedSourceBenchmarker,
    BenchmarkReport,
    generate_summary,
    print_report,
    POLYGLOT_MODULES
)
from datetime import datetime
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Polyglot vs Closed Source Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Ejecutar todos los benchmarks
  python scripts/run_polyglot_benchmarks.py --full
  
  # Ejecutar solo módulos específicos
  python scripts/run_polyglot_benchmarks.py --modules kv_cache compression
  
  # Especificar modelo para inference
  python scripts/run_polyglot_benchmarks.py --full --model gpt2
  
  # Solo benchmarks polyglot (sin closed source)
  python scripts/run_polyglot_benchmarks.py --polyglot-only
  
  # Solo benchmarks closed source
  python scripts/run_polyglot_benchmarks.py --closed-source-only
        """
    )
    
    parser.add_argument(
        "--full",
        action="store_true",
        help="Ejecutar suite completa de benchmarks"
    )
    
    parser.add_argument(
        "--polyglot-only",
        action="store_true",
        help="Ejecutar solo benchmarks polyglot"
    )
    
    parser.add_argument(
        "--closed-source-only",
        action="store_true",
        help="Ejecutar solo benchmarks de modelos closed source"
    )
    
    parser.add_argument(
        "--modules",
        nargs="+",
        choices=["kv_cache", "compression", "attention", "inference", "all"],
        default=["all"],
        help="Módulos específicos a benchmarkear"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ruta o nombre del modelo para inference (ej: gpt2, microsoft/DialoGPT-small)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directorio de salida para reportes (default: benchmark_reports/)"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Modo verbose"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determinar qué ejecutar
    run_polyglot = not args.closed_source_only
    run_closed_source = not args.polyglot_only
    
    if args.full:
        run_polyglot = True
        run_closed_source = True
    
    # Ejecutar benchmarks polyglot
    polyglot_results = []
    if run_polyglot:
        logger.info("=" * 80)
        logger.info("EJECUTANDO BENCHMARKS POLYGLOT")
        logger.info("=" * 80)
        
        benchmarker = PolyglotBenchmarker()
        
        # Determinar qué módulos ejecutar
        modules_to_run = args.modules
        if "all" in modules_to_run:
            modules_to_run = ["kv_cache", "compression", "attention", "inference"]
        
        if "kv_cache" in modules_to_run:
            polyglot_results.extend(benchmarker.benchmark_kv_cache())
        
        if "compression" in modules_to_run:
            polyglot_results.extend(benchmarker.benchmark_compression())
        
        if "attention" in modules_to_run:
            polyglot_results.extend(benchmarker.benchmark_attention())
        
        if "inference" in modules_to_run:
            polyglot_results.extend(benchmarker.benchmark_inference(args.model))
    
    # Ejecutar benchmarks closed source
    closed_source_results = []
    if run_closed_source:
        logger.info("=" * 80)
        logger.info("EJECUTANDO BENCHMARKS CLOSED SOURCE")
        logger.info("=" * 80)
        
        closed_source_benchmarker = ClosedSourceBenchmarker()
        closed_source_results = closed_source_benchmarker.run_all_benchmarks()
    
    # Generar resumen
    summary = generate_summary(polyglot_results, closed_source_results)
    
    # Crear reporte
    report = BenchmarkReport(
        timestamp=datetime.now().isoformat(),
        polyglot_results=polyglot_results,
        closed_source_results=closed_source_results,
        module_status=POLYGLOT_MODULES,
        summary=summary
    )
    
    # Imprimir reporte
    print_report(report)
    
    # Guardar reporte
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(__file__).parent.parent / "benchmark_reports"
    
    output_dir.mkdir(exist_ok=True, parents=True)
    report_path = output_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report.save_json(report_path)
    
    logger.info(f"\n✓ Reporte guardado en: {report_path}")
    
    # Estadísticas finales
    successful_polyglot = sum(1 for r in polyglot_results if r.success)
    successful_closed_source = sum(1 for r in closed_source_results if r.success)
    
    logger.info(f"\nResumen:")
    logger.info(f"  Polyglot: {successful_polyglot}/{len(polyglot_results)} tests exitosos")
    logger.info(f"  Closed Source: {successful_closed_source}/{len(closed_source_results)} tests exitosos")
    
    # Exit code basado en resultados
    if len(polyglot_results) > 0 and successful_polyglot == 0:
        logger.error("Ningún benchmark polyglot fue exitoso")
        sys.exit(1)
    
    if len(polyglot_results) == 0 and run_polyglot:
        logger.error("No se ejecutaron benchmarks polyglot")
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()













