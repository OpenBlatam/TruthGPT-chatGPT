"""
Best Libraries Example - Demonstration of the best optimization libraries
Shows the most advanced and cutting-edge optimization libraries available
"""

import torch
import torch.nn as nn
import logging
import time
import numpy as np
from pathlib import Path

# Import best libraries modules
from ..core import (
    BestLibraries, LibraryCategory, LibraryInfo,
    create_best_libraries, best_libraries_context,
    LibraryRecommender, RecommendationRequest, RecommendationLevel, LibraryRecommendation,
    create_library_recommender, library_recommender_context
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_best_libraries_overview():
    """Example of best libraries overview."""
    print("📚 Best Libraries Overview")
    print("=" * 60)
    
    with best_libraries_context() as libraries:
        # Get library statistics
        stats = libraries.get_library_statistics()
        print(f"📊 Library Statistics:")
        print(f"  Total libraries: {stats['total_libraries']}")
        print(f"  Categories: {stats['categories']}")
        print(f"  Average performance rating: {stats['avg_performance_rating']:.2f}")
        print(f"  Average ease of use: {stats['avg_ease_of_use']:.2f}")
        print(f"  Average documentation quality: {stats['avg_documentation_quality']:.2f}")
        print(f"  Average community support: {stats['avg_community_support']:.2f}")
        
        # Get top performers
        print(f"\n🏆 Top Performers:")
        for i, lib in enumerate(stats['top_performers'], 1):
            print(f"  {i}. {lib}")
        
        # Get easiest to use
        print(f"\n😊 Easiest to Use:")
        for i, lib in enumerate(stats['easiest_to_use'], 1):
            print(f"  {i}. {lib}")
        
        # Get best documented
        print(f"\n📖 Best Documented:")
        for i, lib in enumerate(stats['best_documented'], 1):
            print(f"  {i}. {lib}")

def example_library_categories():
    """Example of library categories."""
    print("\n📂 Library Categories")
    print("=" * 60)
    
    with best_libraries_context() as libraries:
        categories = [
            LibraryCategory.DEEP_LEARNING,
            LibraryCategory.OPTIMIZATION,
            LibraryCategory.SCIENTIFIC,
            LibraryCategory.GPU_ACCELERATION,
            LibraryCategory.DISTRIBUTED,
            LibraryCategory.VISUALIZATION,
            LibraryCategory.MONITORING,
            LibraryCategory.QUANTUM,
            LibraryCategory.AI_ML,
            LibraryCategory.PRODUCTION
        ]
        
        for category in categories:
            print(f"\n🔍 {category.value.upper()} Libraries:")
            libs = libraries.get_libraries_by_category(category)
            
            for lib in libs:
                print(f"  📚 {lib.name} (v{lib.version})")
                print(f"     Performance: {lib.performance_rating}/10")
                print(f"     Ease of use: {lib.ease_of_use}/10")
                print(f"     Documentation: {lib.documentation_quality}/10")
                print(f"     Community: {lib.community_support}/10")
                print(f"     Description: {lib.description}")
                print(f"     Key features: {', '.join(lib.features[:3])}")
                print()

def example_top_libraries():
    """Example of top libraries by category."""
    print("\n🏆 Top Libraries by Category")
    print("=" * 60)
    
    with best_libraries_context() as libraries:
        categories = [
            LibraryCategory.DEEP_LEARNING,
            LibraryCategory.OPTIMIZATION,
            LibraryCategory.SCIENTIFIC,
            LibraryCategory.GPU_ACCELERATION,
            LibraryCategory.DISTRIBUTED,
            LibraryCategory.VISUALIZATION,
            LibraryCategory.MONITORING,
            LibraryCategory.QUANTUM,
            LibraryCategory.AI_ML,
            LibraryCategory.PRODUCTION
        ]
        
        for category in categories:
            print(f"\n🔍 Top {category.value.upper()} Libraries:")
            top_libs = libraries.get_top_libraries(category, limit=3)
            
            for i, lib in enumerate(top_libs, 1):
                print(f"  {i}. {lib.name} - {lib.performance_rating}/10")
                print(f"     {lib.description}")
                print(f"     Features: {', '.join(lib.features[:3])}")
                print(f"     Pros: {', '.join(lib.pros[:2])}")
                print()

def example_library_comparison():
    """Example of library comparison."""
    print("\n⚖️ Library Comparison")
    print("=" * 60)
    
    with best_libraries_context() as libraries:
        # Compare deep learning libraries
        print("🔍 Deep Learning Libraries Comparison:")
        comparison = libraries.compare_libraries(['pytorch', 'tensorflow', 'jax'])
        
        if comparison:
            print(f"  Libraries: {comparison['libraries']}")
            print(f"  Performance ratings: {comparison['performance_ratings']}")
            print(f"  Ease of use: {comparison['ease_of_use']}")
            print(f"  Documentation quality: {comparison['documentation_quality']}")
            print(f"  Community support: {comparison['community_support']}")
            print(f"  Installation difficulty: {comparison['installation_difficulty']}")
        
        print("\n🔍 Optimization Libraries Comparison:")
        comparison = libraries.compare_libraries(['optuna', 'hyperopt', 'scikit-optimize'])
        
        if comparison:
            print(f"  Libraries: {comparison['libraries']}")
            print(f"  Performance ratings: {comparison['performance_ratings']}")
            print(f"  Ease of use: {comparison['ease_of_use']}")
            print(f"  Documentation quality: {comparison['documentation_quality']}")
            print(f"  Community support: {comparison['community_support']}")
            print(f"  Installation difficulty: {comparison['installation_difficulty']}")

def example_library_recommendations():
    """Example of library recommendations."""
    print("\n🎯 Library Recommendations")
    print("=" * 60)
    
    with library_recommender_context() as recommender:
        # Test different use cases
        use_cases = [
            'deep_learning',
            'optimization',
            'scientific',
            'distributed',
            'visualization',
            'monitoring',
            'quantum',
            'ml',
            'production'
        ]
        
        for use_case in use_cases:
            print(f"\n🔍 Recommendations for {use_case.upper()}:")
            
            # Create recommendation request
            request = RecommendationRequest(
                use_case=use_case,
                experience_level=RecommendationLevel.INTERMEDIATE,
                performance_requirements=['high_performance', 'scalability'],
                constraints=['gpu', 'production'],
                preferences=['easy_to_use', 'well_documented'],
                budget='medium',
                timeline='1-3 months',
                team_size=5
            )
            
            # Get recommendations
            recommendations = recommender.recommend_libraries(request)
            
            # Display top 3 recommendations
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"  {i}. {rec.library_name}")
                print(f"     Suitability: {rec.suitability_score:.2f}")
                print(f"     Confidence: {rec.confidence_score:.2f}")
                print(f"     Performance: {rec.performance_score:.2f}")
                print(f"     Ease of use: {rec.ease_of_use_score:.2f}")
                print(f"     Documentation: {rec.documentation_score:.2f}")
                print(f"     Reasoning: {rec.reasoning}")
                print(f"     Pros: {', '.join(rec.pros[:2])}")
                print()

def example_installation_guides():
    """Example of installation guides."""
    print("\n📦 Installation Guides")
    print("=" * 60)
    
    with best_libraries_context() as libraries:
        # Get installation guides for popular libraries
        popular_libraries = ['pytorch', 'tensorflow', 'numpy', 'scipy', 'optuna']
        
        for lib_name in popular_libraries:
            print(f"\n📚 Installation Guide for {lib_name.upper()}:")
            guide = libraries.get_installation_guide(lib_name)
            
            if guide:
                print(f"  Version: {guide['version']}")
                print(f"  Dependencies: {', '.join(guide['dependencies'])}")
                print(f"  Installation commands:")
                for cmd in guide['installation_commands']:
                    print(f"    {cmd}")
                print(f"  Verification commands:")
                for cmd in guide['verification_commands']:
                    print(f"    {cmd}")
                print(f"  Troubleshooting:")
                for tip in guide['troubleshooting']:
                    print(f"    {tip}")

def example_personalized_recommendations():
    """Example of personalized recommendations."""
    print("\n👤 Personalized Recommendations")
    print("=" * 60)
    
    with library_recommender_context() as recommender:
        # Simulate user preferences
        user_id = "user_123"
        preferences = {
            'prefer_performance': True,
            'prefer_ease_of_use': False,
            'prefer_documentation': True,
            'prefer_community_support': True,
            'avoid_complex_installation': True
        }
        
        # Update user preferences
        recommender.update_user_preferences(user_id, preferences)
        
        # Create recommendation request
        request = RecommendationRequest(
            use_case='deep_learning',
            experience_level=RecommendationLevel.ADVANCED,
            performance_requirements=['high_performance', 'gpu_acceleration'],
            constraints=['production', 'scalability'],
            preferences=['performance', 'documentation'],
            budget='high',
            timeline='1 month',
            team_size=10
        )
        
        # Get personalized recommendations
        recommendations = recommender.get_personalized_recommendations(user_id, request)
        
        print(f"🎯 Personalized recommendations for {user_id}:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"  {i}. {rec.library_name}")
            print(f"     Suitability: {rec.suitability_score:.2f}")
            print(f"     Confidence: {rec.confidence_score:.2f}")
            print(f"     Performance: {rec.performance_score:.2f}")
            print(f"     Reasoning: {rec.reasoning}")
            print(f"     Implementation guide:")
            for step in rec.implementation_guide[:3]:
                print(f"       {step}")
            print()

def example_recommendation_statistics():
    """Example of recommendation statistics."""
    print("\n📊 Recommendation Statistics")
    print("=" * 60)
    
    with library_recommender_context() as recommender:
        # Simulate some recommendations
        for i in range(5):
            request = RecommendationRequest(
                use_case='deep_learning',
                experience_level=RecommendationLevel.INTERMEDIATE,
                performance_requirements=['high_performance'],
                constraints=['gpu'],
                preferences=['easy_to_use']
            )
            recommender.recommend_libraries(request)
        
        # Get statistics
        stats = recommender.get_recommendation_statistics()
        
        print(f"📈 Recommendation Statistics:")
        print(f"  Total recommendations: {stats['total_recommendations']}")
        print(f"  Use case distribution: {stats['use_case_distribution']}")
        print(f"  Experience level distribution: {stats['experience_level_distribution']}")
        print(f"  Most recommended libraries: {stats['most_recommended_libraries']}")
        print(f"  Recommendation engine status: {stats['recommendation_engine_status']}")

def example_library_features():
    """Example of library features demonstration."""
    print("\n🔧 Library Features Demonstration")
    print("=" * 60)
    
    with best_libraries_context() as libraries:
        # Demonstrate PyTorch features
        pytorch_info = libraries.get_library('pytorch')
        if pytorch_info:
            print(f"🚀 PyTorch Features:")
            print(f"  Name: {pytorch_info.name}")
            print(f"  Version: {pytorch_info.version}")
            print(f"  Category: {pytorch_info.category.value}")
            print(f"  Performance rating: {pytorch_info.performance_rating}/10")
            print(f"  Ease of use: {pytorch_info.ease_of_use}/10")
            print(f"  Documentation quality: {pytorch_info.documentation_quality}/10")
            print(f"  Community support: {pytorch_info.community_support}/10")
            print(f"  Installation difficulty: {pytorch_info.installation_difficulty}/10")
            print(f"  Description: {pytorch_info.description}")
            print(f"  Features: {', '.join(pytorch_info.features)}")
            print(f"  Use cases: {', '.join(pytorch_info.use_cases)}")
            print(f"  Pros: {', '.join(pytorch_info.pros)}")
            print(f"  Cons: {', '.join(pytorch_info.cons)}")
            print(f"  Alternatives: {', '.join(pytorch_info.alternatives)}")
        
        print("\n🔍 Optuna Features:")
        optuna_info = libraries.get_library('optuna')
        if optuna_info:
            print(f"  Name: {optuna_info.name}")
            print(f"  Version: {optuna_info.version}")
            print(f"  Category: {optuna_info.category.value}")
            print(f"  Performance rating: {optuna_info.performance_rating}/10")
            print(f"  Ease of use: {optuna_info.ease_of_use}/10")
            print(f"  Documentation quality: {optuna_info.documentation_quality}/10")
            print(f"  Community support: {optuna_info.community_support}/10")
            print(f"  Installation difficulty: {optuna_info.installation_difficulty}/10")
            print(f"  Description: {optuna_info.description}")
            print(f"  Features: {', '.join(optuna_info.features)}")
            print(f"  Use cases: {', '.join(optuna_info.use_cases)}")
            print(f"  Pros: {', '.join(optuna_info.pros)}")
            print(f"  Cons: {', '.join(optuna_info.cons)}")
            print(f"  Alternatives: {', '.join(optuna_info.alternatives)}")

def main():
    """Main example function."""
    print("📚 Best Libraries Demonstration")
    print("=" * 70)
    print("Curated collection of the best optimization libraries")
    print("=" * 70)
    
    try:
        # Run all library examples
        example_best_libraries_overview()
        example_library_categories()
        example_top_libraries()
        example_library_comparison()
        example_library_recommendations()
        example_installation_guides()
        example_personalized_recommendations()
        example_recommendation_statistics()
        example_library_features()
        
        print("\n✅ All library examples completed successfully!")
        print("📚 The best libraries are now available!")
        
        print("\n📚 Best Libraries Demonstrated:")
        print("  🚀 Deep Learning:")
        print("    • PyTorch: Dynamic neural networks")
        print("    • TensorFlow: End-to-end ML platform")
        print("    • JAX: High-performance ML research")
        
        print("  🔍 Optimization:")
        print("    • Optuna: Hyperparameter optimization")
        print("    • Hyperopt: Distributed optimization")
        print("    • Scikit-optimize: Bayesian optimization")
        
        print("  🔬 Scientific Computing:")
        print("    • NumPy: Fundamental arrays")
        print("    • SciPy: Scientific computing")
        print("    • CuPy: GPU-accelerated arrays")
        
        print("  🌐 Distributed Computing:")
        print("    • Ray: Distributed framework")
        print("    • Dask: Parallel computing")
        
        print("  📊 Visualization:")
        print("    • Matplotlib: Comprehensive plotting")
        print("    • Plotly: Interactive visualization")
        
        print("  📈 Monitoring:")
        print("    • Weights & Biases: Experiment tracking")
        print("    • TensorBoard: TensorFlow visualization")
        
        print("  🌌 Quantum Computing:")
        print("    • Qiskit: Quantum framework")
        print("    • Cirq: Google Quantum AI")
        
        print("  🤖 AI/ML:")
        print("    • Scikit-learn: Machine learning")
        print("    • XGBoost: Gradient boosting")
        
        print("  🏭 Production:")
        print("    • FastAPI: Modern web framework")
        print("    • MLflow: ML lifecycle management")
        
        print("\n🎯 Key Features:")
        print("  • Comprehensive library database")
        print("  • Intelligent recommendations")
        print("  • Performance ratings")
        print("  • Installation guides")
        print("  • Comparison tools")
        print("  • Personalized suggestions")
        print("  • Community insights")
        
        print("\n📊 Library Statistics:")
        print("  • Total libraries: 20+")
        print("  • Categories: 10")
        print("  • Average performance: 8.5/10")
        print("  • Average ease of use: 8.2/10")
        print("  • Average documentation: 8.8/10")
        print("  • Average community support: 8.9/10")
        
    except Exception as e:
        logger.error(f"Library example failed: {e}")
        print(f"❌ Library example failed: {e}")

if __name__ == "__main__":
    main()




