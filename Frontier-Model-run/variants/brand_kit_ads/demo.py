"""
Brand Kit Ads Model - Interactive Demo

This demo showcases the Brand Kit Ads model capabilities:
1. Website brand analysis and extraction
2. Brand kit visualization
3. Targeted ad content generation
4. Real-time brand-aware content creation
5. Multi-format ad generation (social, email, banner, etc.)
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import streamlit as st
import requests
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
import yaml
import os
import time
from typing import Dict, List, Optional, Any, Tuple
import webcolors
import colorsys
from urllib.parse import urlparse
import base64
from io import BytesIO
import re

from model import BrandKitAdsModel, BrandKitAdsConfig, BrandKitExtraction, AdContent


class BrandKitAdsDemo:
    """Interactive demo for Brand Kit Ads model"""
    
    def __init__(self, config_path: str, model_size: str = "medium"):
        self.config_path = config_path
        self.model_size = model_size
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self._initialize_model()
        
        # Demo state
        self.current_brand_kit = None
        self.current_brand_embedding = None
        self.generation_history = []
        
        # Content templates
        self.content_templates = self._load_content_templates()
        
        # Color palettes for visualization
        self.color_palettes = {
            'tech': ['#1a1a1a', '#ffffff', '#007bff', '#28a745', '#ffc107'],
            'luxury': ['#8b4513', '#daa520', '#f5deb3', '#2f4f4f', '#800080'],
            'playful': ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57'],
            'corporate': ['#2c3e50', '#3498db', '#e74c3c', '#f39c12', '#9b59b6'],
            'creative': ['#6c5ce7', '#a29bfe', '#fd79a8', '#fdcb6e', '#00b894']
        }
    
    def _load_config(self) -> BrandKitAdsConfig:
        """Load model configuration"""
        try:
            config = BrandKitAdsConfig.from_yaml(self.config_path)
            
            # Apply model size variant
            if self.model_size in ['small', 'medium', 'large']:
                with open(self.config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                
                variant_config = yaml_config.get('model_variants', {}).get(self.model_size, {})
                for key, value in variant_config.items():
                    if hasattr(config, key) and key != 'description':
                        setattr(config, key, value)
            
            return config
        except Exception as e:
            st.error(f"Error loading configuration: {e}")
            return BrandKitAdsConfig()
    
    def _initialize_model(self):
        """Initialize model and tokenizer"""
        try:
            # Create model with configuration
            self.model = BrandKitAdsModel(self.config)
            
            # Initialize tokenizer (placeholder - would use actual tokenizer)
            self.tokenizer = self._create_dummy_tokenizer()
            
            # Set to evaluation mode
            self.model.eval()
            
            st.success(f"Model initialized successfully! ({self.model_size} variant)")
            
        except Exception as e:
            st.error(f"Error initializing model: {e}")
            self.model = None
            self.tokenizer = None
    
    def _create_dummy_tokenizer(self):
        """Create dummy tokenizer for demo purposes"""
        class DummyTokenizer:
            def __init__(self):
                self.vocab_size = 50257
                self.pad_token_id = 0
                self.eos_token_id = 1
                
            def encode(self, text, max_length=512, padding=True, truncation=True, return_tensors="pt"):
                # Simple word-based tokenization for demo
                words = text.lower().split()
                token_ids = [hash(word) % self.vocab_size for word in words]
                
                if truncation and len(token_ids) > max_length:
                    token_ids = token_ids[:max_length]
                
                if padding:
                    while len(token_ids) < max_length:
                        token_ids.append(self.pad_token_id)
                
                if return_tensors == "pt":
                    return {
                        'input_ids': torch.tensor([token_ids]),
                        'attention_mask': torch.tensor([[1 if t != self.pad_token_id else 0 for t in token_ids]])
                    }
                return token_ids
            
            def decode(self, token_ids, skip_special_tokens=True):
                # Simple decoding for demo
                return f"Generated content based on {len(token_ids)} tokens"
        
        return DummyTokenizer()
    
    def _load_content_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load content generation templates"""
        return {
            'social_media': {
                'structure': ['hook', 'value_proposition', 'call_to_action'],
                'max_length': 280,
                'tone': 'engaging',
                'examples': [
                    "🚀 Transform your business with cutting-edge AI solutions! Join thousands of satisfied customers. Get started today! #Innovation",
                    "💡 Discover the future of productivity. Our platform delivers 10x faster results. Try it free for 30 days!",
                    "🎯 Ready to boost your sales? Our proven system increases conversions by 300%. Book your demo now!"
                ]
            },
            'email_marketing': {
                'structure': ['subject_line', 'greeting', 'body', 'call_to_action'],
                'max_length': 500,
                'tone': 'professional',
                'examples': [
                    "Subject: Unlock 50% savings this week only\n\nHi [Name],\n\nWe're excited to offer you exclusive access to our biggest sale of the year. Save 50% on all premium features.\n\nClaim your discount now →",
                    "Subject: Your productivity upgrade is here\n\nDear [Name],\n\nTired of juggling multiple tools? Our all-in-one platform streamlines your workflow and saves you 5 hours per week.\n\nStart your free trial →"
                ]
            },
            'banner_ads': {
                'structure': ['headline', 'subheadline', 'call_to_action'],
                'max_length': 100,
                'tone': 'impactful',
                'examples': [
                    "SAVE 70% TODAY\nLimited time offer on premium plans\nSHOP NOW",
                    "NEW ARRIVAL\nRevolutionary AI-powered solution\nLEARN MORE",
                    "FREE TRIAL\n30 days of unlimited access\nSTART NOW"
                ]
            },
            'video_scripts': {
                'structure': ['hook', 'problem', 'solution', 'benefits', 'call_to_action'],
                'max_length': 800,
                'tone': 'conversational',
                'examples': [
                    "[0-3s] Are you tired of spending hours on repetitive tasks?\n[4-10s] Most businesses waste 40% of their time on manual processes.\n[11-20s] Our AI automation platform eliminates the busy work.\n[21-25s] Save time, reduce errors, increase productivity.\n[26-30s] Try it free at example.com"
                ]
            }
        }
    
    def analyze_website_brand(self, url: str) -> Optional[BrandKitExtraction]:
        """Analyze website and extract brand kit"""
        try:
            if not self.model:
                st.error("Model not initialized")
                return None
            
            # Simulate website analysis (in real implementation, this would scrape the website)
            with st.spinner(f"Analyzing website: {url}"):
                time.sleep(2)  # Simulate processing time
                
                # Extract domain for brand personality simulation
                domain = urlparse(url).netloc.lower()
                
                # Simulate brand analysis based on domain
                brand_kit = self._simulate_brand_analysis(domain)
                
                # Store current brand kit
                self.current_brand_kit = brand_kit
                
                # Generate brand embedding (simulated)
                self.current_brand_embedding = torch.randn(self.config.brand_embedding_size)
                
                return brand_kit
                
        except Exception as e:
            st.error(f"Error analyzing website: {e}")
            return None
    
    def _simulate_brand_analysis(self, domain: str) -> BrandKitExtraction:
        """Simulate brand analysis for demo purposes"""
        
        # Determine brand category based on domain keywords
        if any(keyword in domain for keyword in ['tech', 'ai', 'software', 'app']):
            category = 'tech'
        elif any(keyword in domain for keyword in ['luxury', 'premium', 'exclusive']):
            category = 'luxury'
        elif any(keyword in domain for keyword in ['fun', 'game', 'play', 'creative']):
            category = 'playful'
        elif any(keyword in domain for keyword in ['bank', 'finance', 'corporate', 'business']):
            category = 'corporate'
        else:
            category = 'creative'
        
        # Get color palette for category
        colors = self.color_palettes.get(category, self.color_palettes['tech'])
        
        # Create brand kit
        brand_kit = BrandKitExtraction(
            primary_colors=colors[:3],
            secondary_colors=colors[3:5] if len(colors) > 3 else ['#6c757d'],
            accent_colors=['#ffc107', '#dc3545'],
            typography={
                'primary_font': 'Arial, sans-serif' if category == 'tech' else 'Georgia, serif',
                'secondary_font': 'Helvetica, sans-serif',
                'font_sizes': [12, 14, 16, 18, 24, 32, 48],
                'line_heights': [1.2, 1.4, 1.6],
                'font_weights': [400, 600, 700]
            },
            logo_elements=[
                {'type': 'text', 'content': domain.split('.')[0].title()},
                {'type': 'icon', 'style': 'modern'}
            ],
            design_patterns={
                'layout': 'grid' if category == 'tech' else 'flexbox',
                'spacing': 'normal',
                'border_radius': 8 if category == 'modern' else 4,
                'shadows': True,
                'animations': category in ['playful', 'creative']
            },
            brand_personality={
                'professional': 0.9 if category == 'corporate' else 0.7,
                'modern': 0.8 if category == 'tech' else 0.6,
                'trustworthy': 0.9 if category == 'corporate' else 0.8,
                'innovative': 0.9 if category == 'tech' else 0.6,
                'playful': 0.8 if category == 'playful' else 0.3,
                'luxury': 0.9 if category == 'luxury' else 0.2
            },
            visual_hierarchy={
                'header_prominence': 0.9,
                'content_structure': 0.8,
                'call_to_action_visibility': 0.85
            },
            spacing_patterns={
                'margin': 16,
                'padding': 12,
                'gap': 8
            },
            brand_voice={
                'tone': 'professional' if category == 'corporate' else 'friendly',
                'formality': 0.8 if category == 'corporate' else 0.6,
                'friendliness': 0.8 if category == 'playful' else 0.6
            }
        )
        
        return brand_kit
    
    def generate_ad_content(
        self,
        prompt: str,
        content_type: str = "social_media",
        target_audience: str = "general",
        num_variants: int = 3
    ) -> List[AdContent]:
        """Generate ad content based on brand analysis"""
        
        if not self.model or not self.current_brand_embedding:
            st.error("Please analyze a website first")
            return []
        
        try:
            with st.spinner(f"Generating {content_type} content..."):
                time.sleep(1)  # Simulate generation time
                
                # Get template for content type
                template = self.content_templates.get(content_type, self.content_templates['social_media'])
                
                # Generate variants
                ad_contents = []
                for i in range(num_variants):
                    # Simulate content generation
                    ad_content = self._generate_content_variant(
                        prompt, content_type, target_audience, template, i
                    )
                    ad_contents.append(ad_content)
                
                # Store in history
                self.generation_history.append({
                    'timestamp': time.time(),
                    'prompt': prompt,
                    'content_type': content_type,
                    'target_audience': target_audience,
                    'variants': ad_contents
                })
                
                return ad_contents
                
        except Exception as e:
            st.error(f"Error generating content: {e}")
            return []
    
    def _generate_content_variant(
        self,
        prompt: str,
        content_type: str,
        target_audience: str,
        template: Dict[str, Any],
        variant_index: int
    ) -> AdContent:
        """Generate a single content variant"""
        
        # Get examples from template
        examples = template.get('examples', [])
        base_example = examples[variant_index % len(examples)] if examples else ""
        
        # Customize based on brand personality
        if self.current_brand_kit:
            personality = self.current_brand_kit.brand_personality
            
            # Adjust tone based on brand personality
            if personality.get('professional', 0) > 0.8:
                tone_modifier = "professional and trustworthy"
            elif personality.get('playful', 0) > 0.7:
                tone_modifier = "fun and engaging"
            elif personality.get('luxury', 0) > 0.7:
                tone_modifier = "premium and exclusive"
            else:
                tone_modifier = "modern and innovative"
        else:
            tone_modifier = "engaging"
        
        # Generate content based on type
        if content_type == "social_media":
            headline = f"🚀 {prompt} - Variant {variant_index + 1}"
            body = f"Experience the difference with our {tone_modifier} approach. Join thousands of satisfied customers!"
            cta = "Learn More Today"
            
        elif content_type == "email_marketing":
            headline = f"Transform Your Experience - {prompt}"
            body = f"Dear Valued Customer,\n\nDiscover how our {tone_modifier} solution can revolutionize your workflow. Limited time offer!"
            cta = "Claim Your Offer"
            
        elif content_type == "banner_ads":
            headline = f"{prompt.upper()}"
            body = f"Limited time offer"
            cta = "SHOP NOW"
            
        elif content_type == "video_scripts":
            headline = f"Video: {prompt}"
            body = f"[0-5s] Hook: Are you ready to transform your business?\n[6-15s] Problem: Most solutions are complicated.\n[16-25s] Solution: Our {tone_modifier} platform simplifies everything.\n[26-30s] CTA: Try it free today!"
            cta = "Visit our website"
            
        else:
            headline = f"Discover {prompt}"
            body = f"Our {tone_modifier} solution delivers exceptional results."
            cta = "Get Started"
        
        # Calculate brand alignment score
        brand_alignment = 0.85 + (variant_index * 0.05)
        
        return AdContent(
            headline=headline,
            subheadline=f"Experience the power of {tone_modifier} innovation",
            body_text=body,
            call_to_action=cta,
            image_descriptions=[
                f"Hero image showcasing {tone_modifier} design",
                f"Lifestyle image with {target_audience} audience",
                "Close-up of key features and benefits"
            ],
            brand_alignment_score=min(brand_alignment, 1.0),
            target_audience=target_audience,
            content_type=content_type,
            visual_suggestions={
                'color_scheme': 'primary_brand_colors',
                'typography': 'brand_fonts',
                'layout': template.get('tone', 'modern'),
                'imagery_style': f'{tone_modifier}_lifestyle'
            }
        )
    
    def visualize_brand_kit(self, brand_kit: BrandKitExtraction):
        """Create visualizations for brand kit"""
        
        # Color palette visualization
        st.subheader("🎨 Color Palette")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Primary Colors**")
            self._display_color_palette(brand_kit.primary_colors)
        
        with col2:
            st.write("**Secondary Colors**")
            self._display_color_palette(brand_kit.secondary_colors)
        
        with col3:
            st.write("**Accent Colors**")
            self._display_color_palette(brand_kit.accent_colors)
        
        # Brand personality radar chart
        st.subheader("🧠 Brand Personality")
        self._create_personality_radar(brand_kit.brand_personality)
        
        # Typography information
        st.subheader("📝 Typography")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Primary Font**")
            st.code(brand_kit.typography.get('primary_font', 'Arial, sans-serif'))
            
            st.write("**Font Sizes**")
            sizes = brand_kit.typography.get('font_sizes', [12, 16, 24])
            st.write(f"Range: {min(sizes)}px - {max(sizes)}px")
        
        with col2:
            st.write("**Secondary Font**")
            st.code(brand_kit.typography.get('secondary_font', 'Georgia, serif'))
            
            st.write("**Font Weights**")
            weights = brand_kit.typography.get('font_weights', [400, 600, 700])
            st.write(f"Available: {', '.join(map(str, weights))}")
        
        # Design patterns
        st.subheader("🎯 Design Patterns")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Layout Style**")
            st.info(brand_kit.design_patterns.get('layout', 'grid'))
            
            st.write("**Spacing**")
            st.info(brand_kit.design_patterns.get('spacing', 'normal'))
        
        with col2:
            st.write("**Border Radius**")
            st.info(f"{brand_kit.design_patterns.get('border_radius', 4)}px")
            
            st.write("**Shadows**")
            st.info("Enabled" if brand_kit.design_patterns.get('shadows', True) else "Disabled")
    
    def _display_color_palette(self, colors: List[str]):
        """Display color palette with swatches"""
        if not colors:
            st.write("No colors detected")
            return
        
        # Create color swatches
        html_colors = ""
        for color in colors:
            html_colors += f"""
            <div style="
                display: inline-block;
                width: 40px;
                height: 40px;
                background-color: {color};
                margin: 2px;
                border: 1px solid #ccc;
                border-radius: 4px;
            " title="{color}"></div>
            """
        
        st.markdown(html_colors, unsafe_allow_html=True)
        
        # Display hex codes
        for color in colors:
            st.code(color)
    
    def _create_personality_radar(self, personality: Dict[str, float]):
        """Create radar chart for brand personality"""
        
        # Prepare data
        categories = list(personality.keys())
        values = list(personality.values())
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Brand Personality',
            line_color='rgb(0, 123, 255)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_ad_content(self, ad_contents: List[AdContent]):
        """Display generated ad content"""
        
        for i, ad_content in enumerate(ad_contents):
            with st.expander(f"Variant {i + 1} - {ad_content.content_type.replace('_', ' ').title()}", expanded=i == 0):
                
                # Content preview
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("**Headline**")
                    st.markdown(f"### {ad_content.headline}")
                    
                    if ad_content.subheadline:
                        st.write("**Subheadline**")
                        st.write(ad_content.subheadline)
                    
                    st.write("**Body Text**")
                    st.write(ad_content.body_text)
                    
                    st.write("**Call to Action**")
                    st.button(ad_content.call_to_action, key=f"cta_{i}")
                
                with col2:
                    # Metrics
                    st.metric("Brand Alignment", f"{ad_content.brand_alignment_score:.1%}")
                    st.metric("Target Audience", ad_content.target_audience.replace('_', ' ').title())
                    
                    # Visual suggestions
                    st.write("**Visual Suggestions**")
                    for key, value in ad_content.visual_suggestions.items():
                        st.write(f"• {key.replace('_', ' ').title()}: {value}")
                
                # Image descriptions
                if ad_content.image_descriptions:
                    st.write("**Suggested Images**")
                    for j, desc in enumerate(ad_content.image_descriptions):
                        st.write(f"{j + 1}. {desc}")
    
    def run_demo(self):
        """Run the interactive demo"""
        
        st.set_page_config(
            page_title="Brand Kit Ads Model Demo",
            page_icon="🎨",
            layout="wide"
        )
        
        st.title("🎨 Brand Kit Ads Model Demo")
        st.markdown("**Analyze websites, extract brand kits, and generate targeted advertising content**")
        
        # Sidebar configuration
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Model information
            st.subheader("Model Info")
            st.info(f"**Size**: {self.model_size.title()}")
            if self.model:
                param_count = sum(p.numel() for p in self.model.parameters())
                st.info(f"**Parameters**: {param_count:,}")
            
            # Demo options
            st.subheader("Demo Options")
            show_advanced = st.checkbox("Show Advanced Options", value=False)
            auto_generate = st.checkbox("Auto-generate Examples", value=True)
            
            if show_advanced:
                generation_temperature = st.slider("Generation Temperature", 0.1, 2.0, 0.8)
                max_variants = st.slider("Max Variants", 1, 5, 3)
            else:
                generation_temperature = 0.8
                max_variants = 3
        
        # Main interface
        tab1, tab2, tab3, tab4 = st.tabs(["🌐 Website Analysis", "🎨 Brand Kit", "📝 Ad Generation", "📊 Analytics"])
        
        with tab1:
            st.header("Website Brand Analysis")
            
            # URL input
            col1, col2 = st.columns([3, 1])
            with col1:
                url = st.text_input(
                    "Enter website URL:",
                    placeholder="https://example.com",
                    help="Enter a website URL to analyze its brand kit"
                )
            
            with col2:
                analyze_button = st.button("🔍 Analyze", type="primary")
            
            # Example URLs
            if auto_generate:
                st.write("**Try these examples:**")
                example_urls = [
                    "https://apple.com",
                    "https://stripe.com",
                    "https://airbnb.com",
                    "https://spotify.com",
                    "https://netflix.com"
                ]
                
                cols = st.columns(len(example_urls))
                for i, example_url in enumerate(example_urls):
                    with cols[i]:
                        if st.button(example_url.replace("https://", ""), key=f"example_{i}"):
                            url = example_url
                            analyze_button = True
            
            # Analysis results
            if analyze_button and url:
                brand_kit = self.analyze_website_brand(url)
                
                if brand_kit:
                    st.success(f"✅ Successfully analyzed {url}")
                    
                    # Quick overview
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Primary Colors", len(brand_kit.primary_colors))
                    
                    with col2:
                        st.metric("Typography Fonts", len(brand_kit.typography.get('font_sizes', [])))
                    
                    with col3:
                        personality_score = sum(brand_kit.brand_personality.values()) / len(brand_kit.brand_personality)
                        st.metric("Personality Score", f"{personality_score:.2f}")
                    
                    with col4:
                        st.metric("Design Patterns", len(brand_kit.design_patterns))
                    
                    # Store for other tabs
                    st.session_state['current_brand_kit'] = brand_kit
                    st.session_state['analyzed_url'] = url
        
        with tab2:
            st.header("Brand Kit Visualization")
            
            if 'current_brand_kit' in st.session_state:
                brand_kit = st.session_state['current_brand_kit']
                url = st.session_state.get('analyzed_url', 'Unknown')
                
                st.info(f"📊 Brand analysis for: **{url}**")
                
                # Visualize brand kit
                self.visualize_brand_kit(brand_kit)
                
                # Export options
                st.subheader("📤 Export Brand Kit")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📋 Copy JSON"):
                        brand_data = {
                            'url': url,
                            'primary_colors': brand_kit.primary_colors,
                            'typography': brand_kit.typography,
                            'brand_personality': brand_kit.brand_personality
                        }
                        st.code(json.dumps(brand_data, indent=2))
                
                with col2:
                    if st.button("🎨 Generate CSS"):
                        css_code = self._generate_css_from_brand_kit(brand_kit)
                        st.code(css_code, language='css')
                
                with col3:
                    if st.button("📊 Download Report"):
                        st.info("Report download would be implemented here")
            
            else:
                st.warning("⚠️ Please analyze a website first in the Website Analysis tab")
        
        with tab3:
            st.header("Ad Content Generation")
            
            if 'current_brand_kit' in st.session_state:
                brand_kit = st.session_state['current_brand_kit']
                url = st.session_state.get('analyzed_url', 'Unknown')
                
                st.info(f"🎯 Generating ads for: **{url}**")
                
                # Generation parameters
                col1, col2 = st.columns(2)
                
                with col1:
                    content_type = st.selectbox(
                        "Content Type:",
                        options=list(self.content_templates.keys()),
                        format_func=lambda x: x.replace('_', ' ').title()
                    )
                    
                    target_audience = st.selectbox(
                        "Target Audience:",
                        options=['general', 'young_adults', 'professionals', 'families', 'seniors', 'students']
                    )
                
                with col2:
                    prompt = st.text_area(
                        "Content Prompt:",
                        placeholder="Describe what you want to advertise...",
                        height=100
                    )
                    
                    num_variants = st.slider("Number of Variants:", 1, max_variants, 3)
                
                # Generate button
                if st.button("🚀 Generate Ad Content", type="primary"):
                    if prompt:
                        ad_contents = self.generate_ad_content(
                            prompt=prompt,
                            content_type=content_type,
                            target_audience=target_audience,
                            num_variants=num_variants
                        )
                        
                        if ad_contents:
                            st.success(f"✅ Generated {len(ad_contents)} content variants")
                            self.display_ad_content(ad_contents)
                            
                            # Store for analytics
                            st.session_state['last_generation'] = {
                                'prompt': prompt,
                                'content_type': content_type,
                                'target_audience': target_audience,
                                'ad_contents': ad_contents
                            }
                    else:
                        st.warning("⚠️ Please enter a content prompt")
                
                # Content templates
                st.subheader("📝 Content Templates")
                template = self.content_templates.get(content_type, {})
                
                if template.get('examples'):
                    st.write("**Example outputs:**")
                    for i, example in enumerate(template['examples'][:2]):
                        with st.expander(f"Example {i + 1}"):
                            st.write(example)
            
            else:
                st.warning("⚠️ Please analyze a website first in the Website Analysis tab")
        
        with tab4:
            st.header("Analytics & Performance")
            
            # Generation history
            if self.generation_history:
                st.subheader("📈 Generation History")
                
                # Create metrics
                total_generations = len(self.generation_history)
                avg_brand_alignment = np.mean([
                    np.mean([variant.brand_alignment_score for variant in gen['variants']])
                    for gen in self.generation_history
                ])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Generations", total_generations)
                with col2:
                    st.metric("Avg Brand Alignment", f"{avg_brand_alignment:.1%}")
                with col3:
                    st.metric("Content Types", len(set(gen['content_type'] for gen in self.generation_history)))
                
                # History table
                history_data = []
                for gen in self.generation_history[-10:]:  # Last 10 generations
                    history_data.append({
                        'Timestamp': time.strftime('%H:%M:%S', time.localtime(gen['timestamp'])),
                        'Content Type': gen['content_type'].replace('_', ' ').title(),
                        'Target Audience': gen['target_audience'].replace('_', ' ').title(),
                        'Variants': len(gen['variants']),
                        'Avg Alignment': f"{np.mean([v.brand_alignment_score for v in gen['variants']]):.1%}"
                    })
                
                df = pd.DataFrame(history_data)
                st.dataframe(df, use_container_width=True)
            
            else:
                st.info("📊 Generate some content to see analytics here")
            
            # Performance metrics
            st.subheader("⚡ Performance Metrics")
            
            if self.model:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Model Information**")
                    param_count = sum(p.numel() for p in self.model.parameters())
                    st.write(f"• Parameters: {param_count:,}")
                    st.write(f"• Model Size: {self.model_size.title()}")
                    st.write(f"• Brand Embedding Size: {self.config.brand_embedding_size}")
                
                with col2:
                    st.write("**Capabilities**")
                    st.write("• ✅ Website Brand Analysis")
                    st.write("• ✅ Color Extraction")
                    st.write("• ✅ Typography Analysis")
                    st.write("• ✅ Multi-format Ad Generation")
                    st.write("• ✅ Brand-aware Content Creation")
    
    def _generate_css_from_brand_kit(self, brand_kit: BrandKitExtraction) -> str:
        """Generate CSS code from brand kit"""
        
        css_code = f"""/* Brand Kit CSS Variables */
:root {{
  /* Primary Colors */
  --primary-color-1: {brand_kit.primary_colors[0] if brand_kit.primary_colors else '#000000'};
  --primary-color-2: {brand_kit.primary_colors[1] if len(brand_kit.primary_colors) > 1 else '#ffffff'};
  --primary-color-3: {brand_kit.primary_colors[2] if len(brand_kit.primary_colors) > 2 else '#007bff'};
  
  /* Typography */
  --primary-font: {brand_kit.typography.get('primary_font', 'Arial, sans-serif')};
  --secondary-font: {brand_kit.typography.get('secondary_font', 'Georgia, serif')};
  
  /* Spacing */
  --spacing-sm: {brand_kit.spacing_patterns.get('gap', 8)}px;
  --spacing-md: {brand_kit.spacing_patterns.get('padding', 12)}px;
  --spacing-lg: {brand_kit.spacing_patterns.get('margin', 16)}px;
  
  /* Border Radius */
  --border-radius: {brand_kit.design_patterns.get('border_radius', 4)}px;
}}

/* Base Styles */
body {{
  font-family: var(--primary-font);
  color: var(--primary-color-1);
  background-color: var(--primary-color-2);
}}

h1, h2, h3, h4, h5, h6 {{
  font-family: var(--primary-font);
  color: var(--primary-color-1);
}}

.btn-primary {{
  background-color: var(--primary-color-3);
  border-color: var(--primary-color-3);
  border-radius: var(--border-radius);
  padding: var(--spacing-sm) var(--spacing-md);
}}"""
        
        return css_code


def main():
    """Main function to run the demo"""
    
    # Configuration
    config_path = "config.yaml"
    
    # Check if config exists
    if not os.path.exists(config_path):
        st.error(f"Configuration file not found: {config_path}")
        st.info("Please ensure the config.yaml file is in the same directory as this demo.")
        return
    
    # Model size selection
    model_size = st.sidebar.selectbox(
        "Select Model Size:",
        options=['small', 'medium', 'large'],
        index=1,
        help="Choose model size based on your hardware capabilities"
    )
    
    # Initialize and run demo
    try:
        demo = BrandKitAdsDemo(config_path, model_size)
        demo.run_demo()
        
    except Exception as e:
        st.error(f"Error running demo: {e}")
        st.info("Please check your configuration and try again.")


if __name__ == "__main__":
    main()