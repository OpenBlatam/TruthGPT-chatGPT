"""
Test script for brandkit model functionality.
"""

import torch
import yaml
from brandkit.brand_analyzer import create_brand_analyzer_model
from brandkit.content_generator import create_content_generator_model
from brandkit.website_scraper import WebsiteScraper, BrandExtractor

def test_brandkit_models():
    """Test brandkit model instantiation and functionality."""
    print("Brandkit Model Test")
    print("=" * 40)
    
    with open("brandkit/config.yaml", 'r') as f:
        config = yaml.safe_load(f)['brandkit']
    
    print("Testing brand analyzer model instantiation...")
    brand_analyzer = create_brand_analyzer_model(config['brand_analyzer'])
    print(f"✓ Brand analyzer instantiated successfully")
    print(f"  - Model type: {type(brand_analyzer)}")
    print(f"  - Parameters: {sum(p.numel() for p in brand_analyzer.parameters()):,}")
    
    print("\nTesting content generator model instantiation...")
    content_generator = create_content_generator_model(config['content_generator'])
    print(f"✓ Content generator instantiated successfully")
    print(f"  - Model type: {type(content_generator)}")
    print(f"  - Parameters: {sum(p.numel() for p in content_generator.parameters()):,}")
    
    print("\nTesting brand analyzer forward pass...")
    batch_size = 2
    colors = torch.randn(batch_size, 5, 3)
    typography_features = torch.randn(batch_size, 64)
    layout_features = torch.randn(batch_size, 128)
    text_features = torch.randn(batch_size, 10, 768)
    
    with torch.no_grad():
        brand_outputs = brand_analyzer(colors, typography_features, layout_features, text_features)
    
    print(f"✓ Brand analyzer forward pass successful")
    print(f"  - Brand profile shape: {brand_outputs['brand_profile'].shape}")
    print(f"  - Consistency score shape: {brand_outputs['consistency_score'].shape}")
    print(f"  - Visual analysis keys: {list(brand_outputs['visual_analysis'].keys())}")
    print(f"  - Text analysis keys: {list(brand_outputs['text_analysis'].keys())}")
    
    print("\nTesting content generator forward pass...")
    content_type_ids = torch.randint(0, 5, (batch_size,))
    
    with torch.no_grad():
        content_outputs = content_generator(
            brand_outputs['brand_profile'],
            content_type_ids,
            generate_images=True
        )
    
    print(f"✓ Content generator forward pass successful")
    print(f"  - Layout features shape: {content_outputs['layout_features'].shape}")
    print(f"  - Color scheme shape: {content_outputs['color_scheme'].shape}")
    print(f"  - Typography params shape: {content_outputs['typography_params'].shape}")
    print(f"  - Quality score shape: {content_outputs['quality_score'].shape}")
    
    print("\nTesting brand kit extraction...")
    website_data = {
        'colors': colors,
        'typography_features': typography_features,
        'layout_features': layout_features,
        'text_features': text_features
    }
    
    with torch.no_grad():
        brand_kit = brand_analyzer.extract_brand_kit(website_data)
    
    print(f"✓ Brand kit extraction successful")
    print(f"  - Brand kit components: {list(brand_kit.keys())}")
    print(f"  - Color palette size: {len(brand_kit['color_palette'])}")
    print(f"  - Consistency score: {brand_kit['brand_consistency_score']:.3f}")
    
    print("\nTesting content asset generation...")
    content_types = ['social_post', 'logo_variant', 'color_scheme']
    
    with torch.no_grad():
        content_assets = content_generator.generate_content_assets(
            brand_outputs['brand_profile'][0],
            content_types
        )
    
    print(f"✓ Content asset generation successful")
    print(f"  - Generated assets: {list(content_assets.keys())}")
    for asset_type, asset_data in content_assets.items():
        print(f"  - {asset_type}: quality score {asset_data['quality_score']:.3f}")
    
    print("\nTesting website scraper...")
    scraper = WebsiteScraper(config['brand_analyzer'])
    website_data = scraper.scrape_website("https://example.com")
    
    print(f"✓ Website scraper test successful")
    print(f"  - Scraped data keys: {list(website_data.keys())}")
    print(f"  - Title: {website_data['title']}")
    print(f"  - Colors found: {len(website_data['css_styles']['colors'])}")
    
    visual_features = scraper.extract_visual_features(website_data)
    print(f"  - Visual features extracted: {list(visual_features.keys())}")
    print(f"  - Colors shape: {visual_features['colors'].shape}")
    
    print("\nTesting brand extractor...")
    extractor = BrandExtractor(config['brand_analyzer'])
    
    color_palette = extractor.extract_color_palette(visual_features)
    typography_profile = extractor.extract_typography_profile(website_data)
    tone_profile = extractor.extract_tone_profile(
        scraper.extract_text_features(website_data), 
        website_data
    )
    
    print(f"✓ Brand extractor test successful")
    print(f"  - Color palette: {color_palette}")
    print(f"  - Typography fonts: {typography_profile['primary_fonts']}")
    print(f"  - Tone characteristics: {list(tone_profile['voice_characteristics'])}")
    
    brand_analysis = {
        'color_palette': color_palette,
        'typography_profile': typography_profile,
        'tone_profile': tone_profile,
        'consistency_metrics': {'overall_consistency': 0.85}
    }
    
    guidelines = extractor.generate_brand_guidelines(brand_analysis)
    print(f"  - Brand guidelines generated: {list(guidelines.keys())}")
    print(f"  - Primary colors: {guidelines['color_palette']['primary_colors']}")
    
    print("\n" + "=" * 40)
    print("✓ All brandkit tests passed! Implementation is working.")

if __name__ == "__main__":
    test_brandkit_models()
