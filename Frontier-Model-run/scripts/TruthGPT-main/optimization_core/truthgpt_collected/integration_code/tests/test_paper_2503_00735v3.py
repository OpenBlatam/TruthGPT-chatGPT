import unittest
import torch
import sys
import os

# Add the project root to sys.path to allow imports
# Adjusting path to reach the root of the repo 'documentos_blatam'
# File is in truthgpt_collected/integration_code/tests/
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up 4 levels: tests -> integration_code -> truthgpt_collected -> documentos_blatam (root where imports usually start or one level deeper)
# Actually, usually python imports start from the root of the project.
# Let's try adding the directory containing 'truthgpt_collected' to path.
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from truthgpt_collected.integration_code.papers.techniques.paper_2503_00735v3 import (
        Paper2503_00735v3Config,
        EfficientFlashAttention,
        Paper2503_00735v3Module,
        TruthGPT_Paper2503_00735v3_Integration
    )
except ImportError:
    # Try adding one more level up if running from a different context
    project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from truthgpt_collected.integration_code.papers.techniques.paper_2503_00735v3 import (
        Paper2503_00735v3Config,
        EfficientFlashAttention,
        Paper2503_00735v3Module,
        TruthGPT_Paper2503_00735v3_Integration
    )

class TestPaper2503_00735v3(unittest.TestCase):
    def setUp(self):
        self.config = Paper2503_00735v3Config(
            hidden_dim=64,
            num_heads=4,
            chunk_size=16
        )
        self.batch_size = 2
        self.seq_len = 32
        self.device = torch.device("cpu")

    def test_config_defaults(self):
        config = Paper2503_00735v3Config()
        self.assertEqual(config.hidden_dim, 512)
        self.assertTrue(config.use_efficient_attention)
        self.assertTrue(config.use_flash_attention)

    def test_efficient_flash_attention_shapes(self):
        attn = EfficientFlashAttention(
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            chunk_size=self.config.chunk_size
        ).to(self.device)
        
        x = torch.randn(self.batch_size, self.seq_len, self.config.hidden_dim).to(self.device)
        output = attn(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.hidden_dim))

    def test_efficient_flash_attention_metrics(self):
        attn = EfficientFlashAttention(
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            chunk_size=self.config.chunk_size
        ).to(self.device)

        x = torch.randn(self.batch_size, self.seq_len, self.config.hidden_dim).to(self.device)
        _ = attn(x)
        
        metrics = attn.get_metrics()
        self.assertIn('chunk_utilization', metrics)
        self.assertIsInstance(metrics['chunk_utilization'], float)

    def test_module_forward(self):
        module = Paper2503_00735v3Module(self.config).to(self.device)
        x = torch.randn(self.batch_size, self.seq_len, self.config.hidden_dim).to(self.device)
        output = module(x)
        self.assertEqual(output.shape, x.shape)

    def test_module_no_attention(self):
        config = Paper2503_00735v3Config(
            hidden_dim=64,
            use_efficient_attention=False
        )
        module = Paper2503_00735v3Module(config).to(self.device)
        x = torch.randn(self.batch_size, self.seq_len, config.hidden_dim).to(self.device)
        output = module(x)
        self.assertEqual(output.shape, x.shape)
        self.assertIsNone(module.attention)

    def test_integration(self):
        # Mock base model
        class MockBaseModel(torch.nn.Module):
            def forward(self, x):
                return x
        
        base_model = MockBaseModel()
        integration = TruthGPT_Paper2503_00735v3_Integration(base_model, self.config).to(self.device)
        
        x = torch.randn(self.batch_size, self.seq_len, self.config.hidden_dim).to(self.device)
        output = integration(x)
        self.assertEqual(output.shape, x.shape)

    def test_attention_mask(self):
        attn = EfficientFlashAttention(
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            chunk_size=self.config.chunk_size
        ).to(self.device)
        
        x = torch.randn(self.batch_size, self.seq_len, self.config.hidden_dim).to(self.device)
        # Create a dummy mask (batch_size, seq_len)
        mask = torch.ones(self.batch_size, self.seq_len).to(self.device)
        
        # We need to access the chunked_attention method directly to test with mask 
        # because forward() in the class doesn't currently accept a mask argument 
        # (based on the provided file content, forward only takes x)
        
        # Re-create Q, K, V as done in forward to call chunked_attention
        batch_size, seq_len, _ = x.shape
        Q = attn.q_proj(x).view(batch_size, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
        K = attn.k_proj(x).view(batch_size, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
        V = attn.v_proj(x).view(batch_size, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
        
        output = attn.chunked_attention(Q, K, V, attention_mask=mask)
        self.assertEqual(output.shape, (batch_size, attn.num_heads, seq_len, attn.head_dim))

if __name__ == '__main__':
    unittest.main()



