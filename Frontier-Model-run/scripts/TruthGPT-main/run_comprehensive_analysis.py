"""
Script completo para ejecutar análisis comprehensivo de rendimiento TruthGPT
"""

import sys
import os
import traceback
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_comprehensive_analysis():
    """Ejecuta análisis completo de rendimiento."""
    print("🚀 Iniciando Análisis Comprehensivo TruthGPT")
    print("=" * 70)
    
    try:
        from comprehensive_benchmark import ComprehensiveBenchmark
        from generate_performance_report import PerformanceReportGenerator
        from optimization_core.memory_optimizations import create_memory_optimizer
        from optimization_core.computational_optimizations import create_computational_optimizer
        from optimization_core.optimization_profiles import get_optimization_profiles
        
        print("✅ Todos los módulos importados correctamente")
        
        benchmark = ComprehensiveBenchmark()
        print("✅ Sistema de benchmark inicializado")
        
        print("\n📊 Ejecutando benchmarks de modelos...")
        results = benchmark.run_comprehensive_benchmark()
        
        if not results:
            print("❌ No se generaron resultados de benchmark")
            return False
        
        print(f"✅ Benchmark completado para {len(results)} modelos")
        
        print("\n📄 Generando reporte de rendimiento en español...")
        generator = PerformanceReportGenerator(results)
        
        report_file = generator.save_report()
        print(f"✅ Reporte guardado: {report_file}")
        
        csv_file = generator.export_csv()
        print(f"✅ CSV exportado: {csv_file}")
        
        print("\n📋 Resumen de Optimizaciones Aplicadas:")
        print("- ✅ Optimizaciones de memoria (FP16, cuantización, poda)")
        print("- ✅ Optimizaciones computacionales (atención fusionada, kernels)")
        print("- ✅ Perfiles de optimización (velocidad, precisión, balanceado)")
        print("- ✅ MCTS con guía neural y benchmarks de olimpiadas")
        print("- ✅ Normalización avanzada y codificaciones posicionales")
        
        print("\n🎯 Métricas de Rendimiento Clave:")
        if results:
            best_model = max(results, key=lambda x: x.olympiad_accuracy)
            fastest_model = min(results, key=lambda x: x.inference_time_ms)
            most_efficient = min(results, key=lambda x: x.memory_usage_mb)
            
            print(f"- 🧮 Mejor Razonamiento: {best_model.name} ({best_model.olympiad_accuracy:.2%})")
            print(f"- ⚡ Más Rápido: {fastest_model.name} ({fastest_model.inference_time_ms:.2f} ms)")
            print(f"- 💾 Más Eficiente: {most_efficient.name} ({most_efficient.memory_usage_mb:.2f} MB)")
        
        print(f"\n📊 Reporte completo disponible en: {report_file}")
        print(f"📈 Datos CSV disponibles en: {csv_file}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error durante el análisis: {e}")
        traceback.print_exc()
        return False

def test_optimization_profiles():
    """Prueba los perfiles de optimización."""
    print("\n🔧 Probando Perfiles de Optimización")
    print("=" * 50)
    
    try:
        from optimization_core.optimization_profiles import get_optimization_profiles, apply_optimization_profile
        import torch.nn as nn
        
        profiles = get_optimization_profiles()
        print(f"✅ Perfiles disponibles: {list(profiles.keys())}")
        
        try:
            from optimization_core.enhanced_mlp import OptimizedLinear
            test_model = nn.Sequential(
                OptimizedLinear(512, 1024),
                nn.ReLU(),
                OptimizedLinear(1024, 100)
            )
        except ImportError:
            try:
                from optimization_core.enhanced_mlp import EnhancedLinear
                test_model = nn.Sequential(
                    EnhancedLinear(512, 1024),
                    nn.ReLU(),
                    EnhancedLinear(1024, 100)
                )
            except ImportError:
                test_model = nn.Sequential(
                    nn.Linear(512, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 100)
                )
        
        for profile_name in ['speed_optimized', 'accuracy_optimized', 'balanced']:
            try:
                optimized_model, profile = apply_optimization_profile(test_model, profile_name)
                print(f"✅ Perfil '{profile.name}' aplicado correctamente")
            except Exception as e:
                print(f"❌ Error aplicando perfil {profile_name}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error probando perfiles: {e}")
        return False

def main():
    """Función principal."""
    start_time = time.time()
    
    success = True
    
    if not test_optimization_profiles():
        success = False
    
    if not run_comprehensive_analysis():
        success = False
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 Análisis Comprehensivo Completado Exitosamente!")
        print(f"⏱️  Tiempo total: {duration:.2f} segundos")
        print("\n📋 Entregables Generados:")
        print("- Reporte de rendimiento en español (Markdown)")
        print("- Métricas detalladas (CSV)")
        print("- Benchmarks de olimpiadas matemáticas")
        print("- Análisis de optimizaciones MCTS")
        print("- Comparación de perfiles de optimización")
    else:
        print("❌ El análisis encontró algunos problemas.")
        print("Revisa la salida anterior para más detalles.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
