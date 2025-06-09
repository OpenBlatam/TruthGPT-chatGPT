"""
Script to run full comprehensive benchmark and generate performance report
"""

import sys
import os
import traceback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from comprehensive_benchmark import ComprehensiveBenchmark
    from generate_performance_report import PerformanceReportGenerator
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative imports...")
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from comprehensive_benchmark import ComprehensiveBenchmark
    from generate_performance_report import PerformanceReportGenerator

def main():
    """Run comprehensive benchmark and generate report."""
    print("🚀 Iniciando Benchmark Comprehensivo de TruthGPT")
    print("=" * 60)
    
    try:
        benchmark = ComprehensiveBenchmark()
        print("✅ Benchmark inicializado correctamente")
        
        print("\n📊 Ejecutando benchmarks de todos los modelos...")
        results = benchmark.run_comprehensive_benchmark()
        
        if not results:
            print("❌ No se pudieron generar resultados de benchmark")
            return False
        
        print(f"✅ Benchmark completado para {len(results)} modelos")
        
        benchmark.print_summary_report()
        
        json_file = benchmark.save_results()
        print(f"\n💾 Resultados guardados en: {json_file}")
        
        print("\n📄 Generando reporte de rendimiento en español...")
        generator = PerformanceReportGenerator(results)
        
        report_file = generator.save_report()
        print(f"✅ Reporte generado: {report_file}")
        
        csv_file = generator.export_csv()
        print(f"✅ CSV exportado: {csv_file}")
        
        print("\n" + "="*80)
        print("📋 REPORTE DE RENDIMIENTO COMPLETO")
        print("="*80)
        print(generator.generate_spanish_report())
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error durante el benchmark: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
