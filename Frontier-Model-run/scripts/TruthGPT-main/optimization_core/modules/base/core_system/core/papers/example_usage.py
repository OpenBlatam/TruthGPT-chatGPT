"""
Example Usage of Papers Module
===============================

This example demonstrates how to use the Papers module to enhance small models
with research paper techniques.
"""

import torch
import torch.nn as nn
from modules.base.core_system.core.papers import (
    ModelEnhancer,
    PaperRegistry,
    PaperAdapter,
    EnhancementConfig,
    get_paper_registry
)


def example_small_model():
    """Create a simple small model."""
    class SmallModel(nn.Module):
        def __init__(self, d_model=256):
            super().__init__()
            self.embedding = nn.Embedding(1000, d_model)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, nhead=4),
                num_layers=2
            )
            self.output = nn.Linear(d_model, 1000)
        
        def forward(self, x):
            x = self.embedding(x)
            x = self.transformer(x)
            return self.output(x)
    
    return SmallModel()


def example_basic_enhancement():
    """Basic example of enhancing a model."""
    print("=" * 60)
    print("Example 1: Basic Model Enhancement")
    print("=" * 60)
    
    # Create small model
    model = example_small_model()
    print(f"Original model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create enhancer
    enhancer = ModelEnhancer()
    
    # Get suggestions
    suggested = enhancer.suggest_papers(
        model_size="small",
        max_memory_impact="medium"
    )
    print(f"\nSuggested papers: {suggested[:5]}")
    
    # Create enhancement config
    config = EnhancementConfig(paper_ids=suggested[:2])
    
    # Enhance model
    try:
        enhanced_model = enhancer.enhance_model(model, config)
        print(f"✅ Model enhanced with {len(config.paper_ids)} papers")
    except Exception as e:
        print(f"⚠️  Enhancement failed: {e}")


def example_paper_registry():
    """Example of using paper registry."""
    print("\n" + "=" * 60)
    print("Example 2: Using Paper Registry")
    print("=" * 60)
    
    # Get registry
    registry = get_paper_registry()
    
    # List all papers
    papers = registry.list_papers()
    print(f"\nTotal papers available: {len(papers)}")
    
    # Show first 5 papers
    print("\nFirst 5 papers:")
    for paper in papers[:5]:
        print(f"  - {paper.paper_id}: {paper.paper_name}")
        print(f"    Category: {paper.category}, Speedup: {paper.speedup}x")
    
    # Search for speed papers
    speed_papers = registry.search_papers(
        min_speedup=1.5,
        category="techniques"
    )
    print(f"\nPapers with >1.5x speedup in 'techniques': {len(speed_papers)}")
    
    # Get statistics
    stats = registry.get_statistics()
    print(f"\nRegistry statistics:")
    print(f"  Total papers: {stats.get('total_papers', 0)}")
    print(f"  Loaded papers: {stats.get('loaded_papers', 0)}")
    print(f"  Cache hit rate: {stats.get('cache_hit_rate', 0):.2%}")


def example_paper_adapter():
    """Example of using paper adapter."""
    print("\n" + "=" * 60)
    print("Example 3: Using Paper Adapter")
    print("=" * 60)
    
    # Create adapter
    adapter = PaperAdapter()
    
    # List available papers
    available = adapter.list_available_papers(category="techniques")
    print(f"\nAvailable technique papers: {len(available)}")
    print(f"First 3: {available[:3]}")
    
    # Get paper info
    if available:
        paper_id = available[0]
        info = adapter.get_paper_info(paper_id)
        if info:
            print(f"\nPaper info for {paper_id}:")
            print(f"  Name: {info.get('paper_name', 'N/A')}")
            print(f"  Category: {info.get('category', 'N/A')}")
            print(f"  Speedup: {info.get('speedup', 'N/A')}x")
            print(f"  Memory impact: {info.get('memory_impact', 'N/A')}")


def example_enhancement_plan():
    """Example of creating an enhancement plan."""
    print("\n" + "=" * 60)
    print("Example 4: Creating Enhancement Plan")
    print("=" * 60)
    
    # Create enhancer
    enhancer = ModelEnhancer()
    
    # Create plan for speed and accuracy
    plan = enhancer.create_enhancement_plan(
        model_size="small",
        goals=["speed", "accuracy"]
    )
    
    print(f"\nEnhancement plan created:")
    print(f"  Papers to apply: {plan.paper_ids}")
    print(f"  Apply sequentially: {plan.apply_sequentially}")
    print(f"  Preserve original: {plan.preserve_original}")
    
    # Apply plan
    model = example_small_model()
    try:
        enhanced = enhancer.enhance_model(model, plan)
        print(f"\n✅ Model enhanced with plan")
    except Exception as e:
        print(f"\n⚠️  Enhancement failed: {e}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Papers Module - Example Usage")
    print("=" * 60)
    print("\nThis demonstrates how small models can be enhanced with")
    print("research paper techniques integrated into the framework.\n")
    
    try:
        example_basic_enhancement()
        example_paper_registry()
        example_paper_adapter()
        example_enhancement_plan()
        
        print("\n" + "=" * 60)
        print("Examples completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n⚠️  Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()





