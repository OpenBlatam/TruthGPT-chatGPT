#!/usr/bin/env python3
"""
Unit tests for paper_2506_10987v1_chain_of_draft.py
"""

import unittest
import torch
import logging
import sys
import os

# Add current directory to path so we can import the module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from paper_2506_10987v1_chain_of_draft import (
    ChainOfDraftConfig,
    ChainOfDraftModule,
    ChainOfDraftModuleWrapper
)

logging.basicConfig(level=logging.INFO)

class TestChainOfDraft(unittest.TestCase):
    def setUp(self):
        self.config = ChainOfDraftConfig(
            hidden_dim=32,  # Smaller dim for faster tests
            max_words_per_step=5,
            cod_variant="baseline"
        )
        self.module = ChainOfDraftModule(self.config)
        self.batch_size = 2
        self.seq_len = 20
        self.hidden_dim = 32

    def test_config_defaults(self):
        """Test default configuration values."""
        config = ChainOfDraftConfig()
        self.assertEqual(config.hidden_dim, 512)
        self.assertEqual(config.max_words_per_step, 5)
        self.assertEqual(config.cod_variant, "baseline")
        self.assertTrue(config.use_extreme_conciseness)

    def test_draft_encoder_shape(self):
        """Test draft encoder output shape."""
        # Input: [batch, hidden_dim]
        x = torch.randn(self.batch_size, self.hidden_dim)
        output = self.module.encode_draft_step(x)
        self.assertEqual(output.shape, (self.batch_size, self.hidden_dim))
        
        # Input: [batch, 1, hidden_dim]
        x_3d = torch.randn(self.batch_size, 1, self.hidden_dim)
        output_3d = self.module.encode_draft_step(x_3d)
        self.assertEqual(output_3d.shape, (self.batch_size, self.hidden_dim))

    def test_forward_pass_shapes(self):
        """Test full forward pass output shapes."""
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        
        # Test with auto-determined steps
        output, metadata = self.module(x)
        self.assertEqual(output.shape, x.shape)
        self.assertIn('num_draft_steps', metadata)
        
        # Test with fixed steps
        num_steps = 3
        output, metadata = self.module(x, num_draft_steps=num_steps)
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(metadata['num_draft_steps'], num_steps)
        self.assertEqual(metadata['draft_tokens'], num_steps)

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        # Wrong dimensions (2D instead of 3D for forward)
        x_2d = torch.randn(self.batch_size, self.hidden_dim)
        with self.assertRaises(ValueError):
            self.module(x_2d)
            
        # Wrong hidden dim
        x_wrong_dim = torch.randn(self.batch_size, self.seq_len, self.hidden_dim + 10)
        with self.assertRaises(ValueError):
            self.module(x_wrong_dim)

    def test_conciseness_mechanism(self):
        """Test that conciseness mechanism is working (no errors)."""
        x = torch.randn(self.batch_size, 1, self.hidden_dim)
        # Just ensure it runs without error and returns valid tensor
        output = self.module.encode_draft_step(x)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_metrics(self):
        """Test that metrics return expected structure and paper values."""
        metrics = self.module.get_metrics()
        
        # Check exact paper values
        self.assertAlmostEqual(metrics['token_efficiency'], 0.554, places=3)
        self.assertAlmostEqual(metrics['latency_efficiency'], 0.609, places=3)
        self.assertAlmostEqual(metrics['quality_retention'], 0.90, places=2)
        
        # Check dynamic metrics exist
        self.assertIn('avg_words_per_step', metrics)
        self.assertIn('conciseness_score', metrics)

    def test_wrapper(self):
        """Test the wrapper class."""
        wrapper = ChainOfDraftModuleWrapper(self.config)
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        output = wrapper(x)
        self.assertEqual(output.shape, x.shape)
        
        metrics = wrapper.get_metrics()
        self.assertIn('token_efficiency', metrics)

if __name__ == '__main__':
    unittest.main()



