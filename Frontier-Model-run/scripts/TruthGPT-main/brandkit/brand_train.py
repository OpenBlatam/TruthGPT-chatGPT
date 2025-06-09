"""
Training script for brandkit model using GRPO framework.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
try:
    import yaml
except ImportError:
    yaml = None
import logging
import warnings

from .brand_analyzer import create_brand_analyzer_model, BrandAnalyzer
from .content_generator import create_content_generator_model, ContentGenerator
from .website_scraper import WebsiteScraper, BrandExtractor

@dataclass
class BrandKitConfig:
    """Configuration for brandkit training."""
    brand_analyzer: Dict[str, Any]
    content_generator: Dict[str, Any]
    
    learning_rate: float = 1e-4
    batch_size: int = 8
    num_epochs: int = 10
    warmup_steps: int = 1000
    
    max_websites_per_batch: int = 5
    min_brand_consistency: float = 0.7
    
    output_dir: str = "./brandkit_outputs"
    save_interval: int = 1000

@dataclass
class BrandKitTrainingArgs:
    """Training arguments for brandkit model."""
    config_path: str = "brandkit/config.yaml"
    output_dir: str = "./brandkit_outputs"
    logging_level: str = "INFO"
    device: str = "auto"
    
    num_mock_websites: int = 100
    mock_content_types: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.mock_content_types is None:
            self.mock_content_types = [
                'social_post', 'blog_header', 'advertisement', 
                'logo_variant', 'color_scheme'
            ]

class BrandKitTrainer:
    """Trainer for brandkit models using GRPO framework."""
    
    def __init__(self, config: BrandKitConfig, args: BrandKitTrainingArgs):
        self.config = config
        self.args = args
        
        if args.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(args.device)
        
        self.brand_analyzer = create_brand_analyzer_model(config.brand_analyzer)
        self.content_generator = create_content_generator_model(config.content_generator)
        
        self.brand_analyzer.to(self.device)
        self.content_generator.to(self.device)
        
        self.website_scraper = WebsiteScraper(config.brand_analyzer)
        self.brand_extractor = BrandExtractor(config.brand_analyzer)
        
        logging.basicConfig(level=getattr(logging, args.logging_level))
        self.logger = logging.getLogger(__name__)
        
    def create_mock_dataset(self) -> List[Dict[str, Any]]:
        """Create mock dataset for training."""
        warnings.warn("Using mock dataset. Replace with real website data for production.")
        
        mock_websites = []
        for i in range(self.args.num_mock_websites):
            website_data = {
                'url': f'https://example-{i}.com',
                'visual_features': {
                    'colors': torch.randn(1, 5, 3),
                    'typography_features': torch.randn(1, 64),
                    'layout_features': torch.randn(1, 128)
                },
                'text_features': torch.randn(1, 10, 768),
                'brand_consistency_score': torch.rand(1).item(),
                'content_types': self.args.mock_content_types
            }
            mock_websites.append(website_data)
        
        return mock_websites
    
    def process_website_batch(self, websites: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Process a batch of websites for training."""
        batch_data = {
            'colors': [],
            'typography_features': [],
            'layout_features': [],
            'text_features': [],
            'brand_profiles': [],
            'content_assets': []
        }
        
        for website in websites:
            visual_features = website['visual_features']
            batch_data['colors'].append(visual_features['colors'])
            batch_data['typography_features'].append(visual_features['typography_features'])
            batch_data['layout_features'].append(visual_features['layout_features'])
            
            batch_data['text_features'].append(website['text_features'])
        
        for key in ['colors', 'typography_features', 'layout_features', 'text_features']:
            batch_data[key] = torch.cat(batch_data[key], dim=0).to(self.device)
        
        return batch_data
    
    def train_step(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step."""
        brand_outputs = self.brand_analyzer(
            batch_data['colors'],
            batch_data['typography_features'],
            batch_data['layout_features'],
            batch_data['text_features']
        )
        
        batch_size = batch_data['colors'].shape[0]
        content_type_ids = torch.randint(0, 5, (batch_size,)).to(self.device)
        
        content_outputs = self.content_generator(
            brand_outputs['brand_profile'],
            content_type_ids,
            generate_images=True
        )
        
        consistency_loss = nn.MSELoss()(
            brand_outputs['consistency_score'],
            torch.ones_like(brand_outputs['consistency_score'])
        )
        
        quality_loss = nn.MSELoss()(
            content_outputs['quality_score'],
            torch.ones_like(content_outputs['quality_score'])
        )
        
        brand_reg_loss = torch.mean(torch.norm(brand_outputs['brand_profile'], dim=-1))
        
        total_loss = consistency_loss + quality_loss + 0.01 * brand_reg_loss
        
        return {
            'total_loss': total_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'quality_loss': quality_loss.item(),
            'brand_reg_loss': brand_reg_loss.item()
        }
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting brandkit model training...")
        
        dataset = self.create_mock_dataset()
        self.logger.info(f"Created mock dataset with {len(dataset)} websites")
        
        self.brand_analyzer.train()
        self.content_generator.train()
        
        analyzer_optimizer = torch.optim.AdamW(
            self.brand_analyzer.parameters(),
            lr=self.config.learning_rate
        )
        generator_optimizer = torch.optim.AdamW(
            self.content_generator.parameters(),
            lr=self.config.learning_rate
        )
        
        step = 0
        for epoch in range(self.config.num_epochs):
            self.logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")
            
            for i in range(0, len(dataset), self.config.batch_size):
                batch_websites = dataset[i:i + self.config.batch_size]
                batch_data = self.process_website_batch(batch_websites)
                
                losses = self.train_step(batch_data)
                
                analyzer_optimizer.zero_grad()
                generator_optimizer.zero_grad()
                
                total_loss = torch.tensor(losses['total_loss'], requires_grad=True)
                total_loss.backward()
                
                analyzer_optimizer.step()
                generator_optimizer.step()
                
                step += 1
                
                if step % 10 == 0:
                    self.logger.info(f"Step {step}: {losses}")
                
                if step % self.config.save_interval == 0:
                    self.save_checkpoint(step)
        
        self.logger.info("Training completed!")
    
    def save_checkpoint(self, step: int):
        """Save model checkpoint."""
        checkpoint = {
            'step': step,
            'brand_analyzer_state_dict': self.brand_analyzer.state_dict(),
            'content_generator_state_dict': self.content_generator.state_dict(),
            'config': self.config
        }
        
        checkpoint_path = f"{self.args.output_dir}/checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def test_brand_analysis(self, test_url: str = "https://example.com"):
        """Test brand analysis on a single website."""
        self.logger.info(f"Testing brand analysis on {test_url}")
        
        self.brand_analyzer.eval()
        self.content_generator.eval()
        
        with torch.no_grad():
            website_data = self.website_scraper.scrape_website(test_url)
            visual_features = self.website_scraper.extract_visual_features(website_data)
            text_features = self.website_scraper.extract_text_features(website_data)
            
            brand_outputs = self.brand_analyzer(
                visual_features['colors'].to(self.device),
                visual_features['typography_features'].to(self.device),
                visual_features['layout_features'].to(self.device),
                text_features.to(self.device)
            )
            
            brand_kit = self.brand_analyzer.extract_brand_kit({
                'colors': visual_features['colors'],
                'typography_features': visual_features['typography_features'],
                'layout_features': visual_features['layout_features'],
                'text_features': text_features
            })
            
            content_assets = self.content_generator.generate_content_assets(
                brand_outputs['brand_profile'][0],
                ['social_post', 'logo_variant', 'color_scheme']
            )
            
            self.logger.info("Brand analysis results:")
            self.logger.info(f"  Consistency score: {brand_outputs['consistency_score'].item():.3f}")
            self.logger.info(f"  Brand kit components: {list(brand_kit.keys())}")
            self.logger.info(f"  Generated assets: {list(content_assets.keys())}")
            
            return {
                'brand_kit': brand_kit,
                'content_assets': content_assets,
                'consistency_score': brand_outputs['consistency_score'].item()
            }

def main():
    """Main training function."""
    if yaml is None:
        print("YAML not available, using default config")
        config = BrandKitConfig(
            brand_analyzer={
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
            content_generator={
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
        )
    else:
        with open("brandkit/config.yaml", 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = BrandKitConfig(
            brand_analyzer=config_dict['brandkit']['brand_analyzer'],
            content_generator=config_dict['brandkit']['content_generator']
        )
    
    args = BrandKitTrainingArgs()
    
    trainer = BrandKitTrainer(config, args)
    
    print("Testing brand analysis...")
    results = trainer.test_brand_analysis()
    
    print(f"✓ Brand analysis test successful")
    print(f"  Consistency score: {results['consistency_score']:.3f}")
    print(f"  Brand kit extracted: {len(results['brand_kit'])} components")
    print(f"  Content assets generated: {len(results['content_assets'])} types")

if __name__ == "__main__":
    main()
