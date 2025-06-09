"""
Demo script for brandkit functionality.
"""

import torch
from brand_analyzer import create_brand_analyzer_model
from content_generator import create_content_generator_model
from website_scraper import WebsiteScraper, BrandExtractor

def demo_brandkit():
    """Demonstrate brandkit functionality."""
    print("🎨 Brandkit Demo - Website Brand Analysis & Content Generation")
    print("=" * 60)
    
    config = {
        'brand_analyzer': {
            'hidden_size': 768,
            'num_layers': 8,
            'num_attention_heads': 12,
            'dropout': 0.1,
            'max_sequence_length': 2048,
            'color_palette_size': 16,
            'typography_features': 64,
            'layout_features': 128,
            'tone_categories': 10,
            'sentiment_dim': 32,
            'style_dim': 64,
            'visual_feature_dim': 1024,
            'text_feature_dim': 768,
            'metadata_feature_dim': 256
        },
        'content_generator': {
            'hidden_size': 768,
            'num_layers': 6,
            'num_attention_heads': 12,
            'dropout': 0.1,
            'max_sequence_length': 1024,
            'text_vocab_size': 50000,
            'image_feature_dim': 512,
            'layout_dim': 256,
            'brand_profile_dim': 768,
            'style_conditioning_dim': 128,
            'content_types': ['social_post', 'blog_header', 'advertisement', 'logo_variant', 'color_scheme']
        }
    }
    
    print("🔧 Initializing models...")
    brand_analyzer = create_brand_analyzer_model(config['brand_analyzer'])
    content_generator = create_content_generator_model(config['content_generator'])
    scraper = WebsiteScraper(config['brand_analyzer'])
    extractor = BrandExtractor(config['brand_analyzer'])
    
    print(f"✓ Brand Analyzer: {sum(p.numel() for p in brand_analyzer.parameters()):,} parameters")
    print(f"✓ Content Generator: {sum(p.numel() for p in content_generator.parameters()):,} parameters")
    
    print("\n🌐 Analyzing website: https://example-tech-startup.com")
    website_data = scraper.scrape_website("https://example-tech-startup.com")
    
    print(f"📄 Website Title: {website_data['title']}")
    print(f"📝 Description: {website_data['description']}")
    print(f"🎨 Colors Found: {len(website_data['css_styles']['colors'])}")
    print(f"🔤 Fonts Found: {len(website_data['css_styles']['fonts'])}")
    
    print("\n🔍 Extracting visual and text features...")
    visual_features = scraper.extract_visual_features(website_data)
    text_features = scraper.extract_text_features(website_data)
    
    print(f"✓ Visual features: {visual_features['colors'].shape}")
    print(f"✓ Text features: {text_features.shape}")
    
    print("\n🧠 Analyzing brand characteristics...")
    with torch.no_grad():
        brand_outputs = brand_analyzer(
            visual_features['colors'],
            visual_features['typography_features'],
            visual_features['layout_features'],
            text_features
        )
    
    consistency_score = brand_outputs['consistency_score'].item()
    print(f"📊 Brand Consistency Score: {consistency_score:.3f}")
    
    print("\n🎨 Extracting brand kit...")
    brand_kit = brand_analyzer.extract_brand_kit({
        'colors': visual_features['colors'],
        'typography_features': visual_features['typography_features'],
        'layout_features': visual_features['layout_features'],
        'text_features': text_features
    })
    
    print("🎨 Brand Kit Components:")
    print(f"  • Color Palette: {len(brand_kit['color_palette'])} dominant colors")
    print(f"  • Typography Profile: {brand_kit['typography_profile'].shape}")
    print(f"  • Tone Profile: {brand_kit['tone_profile']['dominant_tone']}")
    print(f"  • Style Embedding: {brand_kit['style_embedding'].shape}")
    
    print("\n🚀 Generating brand-consistent content assets...")
    content_types = ['social_post', 'logo_variant', 'color_scheme', 'blog_header', 'advertisement']
    
    with torch.no_grad():
        content_assets = content_generator.generate_content_assets(
            brand_outputs['brand_profile'][0],
            content_types
        )
    
    print("📦 Generated Content Assets:")
    for asset_type, asset_data in content_assets.items():
        quality = asset_data['quality_score']
        color_scheme = asset_data['color_scheme']
        print(f"  • {asset_type.replace('_', ' ').title()}:")
        print(f"    - Quality Score: {quality:.3f}")
        print(f"    - Color Scheme: {color_scheme.shape} (RGB values)")
        print(f"    - Typography Params: {asset_data['typography_params'].shape}")
    
    print("\n📋 Generating comprehensive brand guidelines...")
    color_palette = extractor.extract_color_palette(visual_features)
    typography_profile = extractor.extract_typography_profile(website_data)
    tone_profile = extractor.extract_tone_profile(text_features, website_data)
    
    brand_analysis = {
        'color_palette': color_palette,
        'typography_profile': typography_profile,
        'tone_profile': tone_profile,
        'consistency_metrics': {'overall_consistency': consistency_score}
    }
    
    guidelines = extractor.generate_brand_guidelines(brand_analysis)
    
    print("📖 Brand Guidelines Generated:")
    print(f"  🎨 Primary Colors: {guidelines['color_palette']['primary_colors']}")
    print(f"  🔤 Primary Fonts: {guidelines['typography']['primary_fonts']}")
    print(f"  📝 Voice Tone: {guidelines['voice_tone']['primary_characteristics']}")
    print(f"  📏 Logo Guidelines: {len(guidelines['logo_guidelines']['variations'])} variations")
    print(f"  📐 Layout Principles: {len(guidelines['layout_guidelines']['principles'])} principles")
    
    print("\n" + "=" * 60)
    print("✨ Brandkit Demo Complete!")
    print(f"🎯 Successfully analyzed website and generated {len(content_assets)} content assets")
    print(f"📊 Brand consistency score: {consistency_score:.1%}")
    print("🚀 Ready for brand-consistent content creation!")

if __name__ == "__main__":
    demo_brandkit()
