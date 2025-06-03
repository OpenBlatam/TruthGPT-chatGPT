"""
Brand Kit Ads Model - Native Implementation for Website Brand Analysis and Ad Generation

This package provides a complete solution for:
1. Website brand analysis and extraction
2. Multi-modal brand understanding (visual + textual)
3. Targeted advertising content generation
4. Brand-aware content creation across multiple formats

Key Components:
- BrandKitAdsModel: Main model for brand analysis and ad generation
- BrandKitAdsConfig: Configuration management
- BrandKitAdsTrainer: Advanced training pipeline
- BrandKitExtraction: Brand information data structure
- AdContent: Generated advertising content structure

Usage:
    from brand_kit_ads import BrandKitAdsModel, BrandKitAdsConfig
    
    # Load model
    config = BrandKitAdsConfig.from_yaml("config.yaml")
    model = BrandKitAdsModel(config)
    
    # Analyze website
    brand_kit = model.extract_brand_kit_from_url("https://example.com")
    
    # Generate ads
    ads = model.generate_ad_content(
        brand_embedding=brand_analysis['brand_embedding'],
        prompt="Create social media content for new product launch",
        content_type="social_media"
    )
"""

from .model import (
    BrandKitAdsModel,
    BrandKitAdsConfig,
    BrandKitExtraction,
    AdContent,
    VisionTransformer,
    ColorAnalyzer,
    TypographyAnalyzer,
    DesignPatternAnalyzer,
    BrandEmbedding,
    MultiModalFusion,
    AdContentGenerator,
    WebsiteBrandExtractor
)

from .trainer import (
    BrandKitAdsTrainer,
    BrandKitTrainingArguments,
    WebsiteDataset,
    BrandKitEvaluator,
    BrandConsistencyLoss,
    VisualAlignmentLoss,
    ContentQualityLoss
)

from .demo import BrandKitAdsDemo

__version__ = "1.0.0"
__author__ = "TruthGPT Team"
__email__ = "team@truthgpt.ai"
__description__ = "Native Brand Kit Ads Model for Website Analysis and Ad Generation"

__all__ = [
    # Core model components
    "BrandKitAdsModel",
    "BrandKitAdsConfig",
    "BrandKitExtraction",
    "AdContent",
    
    # Model architecture components
    "VisionTransformer",
    "ColorAnalyzer", 
    "TypographyAnalyzer",
    "DesignPatternAnalyzer",
    "BrandEmbedding",
    "MultiModalFusion",
    "AdContentGenerator",
    "WebsiteBrandExtractor",
    
    # Training components
    "BrandKitAdsTrainer",
    "BrandKitTrainingArguments",
    "WebsiteDataset",
    "BrandKitEvaluator",
    "BrandConsistencyLoss",
    "VisualAlignmentLoss",
    "ContentQualityLoss",
    
    # Demo and utilities
    "BrandKitAdsDemo"
]

# Package metadata
__package_info__ = {
    "name": "brand_kit_ads",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "email": __email__,
    "url": "https://github.com/OpenBlatam/TruthGPT-chatGPT",
    "license": "MIT",
    "keywords": [
        "ai", "machine learning", "brand analysis", "advertising", 
        "content generation", "computer vision", "nlp", "multimodal"
    ],
    "classifiers": [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ]
}

# Model capabilities
CAPABILITIES = {
    "brand_analysis": {
        "color_extraction": "Extract dominant colors from website screenshots",
        "typography_analysis": "Analyze font styles, weights, and hierarchy",
        "design_pattern_recognition": "Identify layout patterns and visual styles",
        "brand_personality_assessment": "Determine brand personality dimensions",
        "visual_hierarchy_analysis": "Understand content structure and emphasis"
    },
    "content_generation": {
        "social_media_ads": "Generate engaging social media content",
        "email_marketing": "Create professional email campaigns",
        "banner_advertisements": "Design impactful banner ad copy",
        "video_scripts": "Write compelling video ad scripts",
        "blog_content": "Generate brand-aligned blog posts",
        "landing_page_copy": "Create conversion-optimized landing pages"
    },
    "brand_consistency": {
        "color_alignment": "Ensure generated content matches brand colors",
        "tone_matching": "Maintain consistent brand voice and tone",
        "style_coherence": "Keep visual and textual style consistent",
        "personality_alignment": "Align content with brand personality traits"
    },
    "multi_modal_understanding": {
        "vision_language_fusion": "Combine visual and textual brand signals",
        "cross_modal_attention": "Attend to relevant visual elements for text generation",
        "brand_embedding_learning": "Learn unified brand representations",
        "contextual_adaptation": "Adapt content based on visual brand context"
    }
}

# Supported content types
CONTENT_TYPES = [
    "social_media",
    "email_marketing", 
    "banner_ads",
    "video_scripts",
    "blog_posts",
    "newsletters",
    "landing_pages",
    "display_ads",
    "search_ads",
    "native_ads"
]

# Supported target audiences
TARGET_AUDIENCES = [
    "general",
    "young_adults",
    "professionals", 
    "families",
    "seniors",
    "students",
    "entrepreneurs",
    "creatives",
    "tech_enthusiasts",
    "health_conscious",
    "budget_conscious",
    "luxury_seekers"
]

# Brand categories
BRAND_CATEGORIES = [
    "technology",
    "fashion",
    "food_beverage",
    "finance",
    "healthcare",
    "education",
    "entertainment",
    "automotive",
    "real_estate",
    "travel",
    "sports",
    "beauty",
    "home_garden",
    "business_services",
    "non_profit"
]

def get_model_info():
    """Get comprehensive model information"""
    return {
        "package_info": __package_info__,
        "capabilities": CAPABILITIES,
        "content_types": CONTENT_TYPES,
        "target_audiences": TARGET_AUDIENCES,
        "brand_categories": BRAND_CATEGORIES,
        "version": __version__
    }

def create_model(config_path: str = None, model_size: str = "medium", **kwargs):
    """
    Convenience function to create a Brand Kit Ads model
    
    Args:
        config_path: Path to configuration file
        model_size: Model size variant ('small', 'medium', 'large')
        **kwargs: Additional configuration parameters
    
    Returns:
        BrandKitAdsModel: Initialized model instance
    """
    if config_path:
        config = BrandKitAdsConfig.from_yaml(config_path)
    else:
        config = BrandKitAdsConfig(**kwargs)
    
    # Apply model size variant if specified
    if model_size in ['small', 'medium', 'large']:
        # This would load variant-specific parameters
        # For now, just use the base config
        pass
    
    return BrandKitAdsModel(config)

def create_trainer(
    model: BrandKitAdsModel,
    train_dataset,
    eval_dataset=None,
    **training_args
):
    """
    Convenience function to create a trainer
    
    Args:
        model: BrandKitAdsModel instance
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        **training_args: Training arguments
    
    Returns:
        BrandKitAdsTrainer: Configured trainer instance
    """
    args = BrandKitTrainingArguments(**training_args)
    return BrandKitAdsTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

def run_demo(config_path: str = "config.yaml", model_size: str = "medium"):
    """
    Run the interactive demo
    
    Args:
        config_path: Path to configuration file
        model_size: Model size variant
    """
    demo = BrandKitAdsDemo(config_path, model_size)
    demo.run_demo()

# Version check
def check_dependencies():
    """Check if all required dependencies are installed"""
    import importlib
    
    required_packages = [
        'torch', 'transformers', 'accelerate', 'PIL', 'cv2',
        'numpy', 'pandas', 'sklearn', 'matplotlib', 'requests'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Warning: Missing required packages: {missing_packages}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All required dependencies are installed")
    return True

# Initialize package
def __init_package__():
    """Initialize package and check dependencies"""
    try:
        check_dependencies()
    except Exception as e:
        print(f"Warning: Could not check dependencies: {e}")

# Run initialization
__init_package__()