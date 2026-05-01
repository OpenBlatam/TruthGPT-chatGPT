"""
MoE Architecture Verification
==============================

Quick verification that the new unified Feed Forward architecture
imports correctly and runs without errors.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))


class TestUnifiedArchitecture(unittest.TestCase):
    """Verify the new unified Feed Forward architecture."""

    def test_import_experts(self):
        """Experts package should import cleanly."""
        from optimization_core.modules.feed_forward.experts import BaseExpert, ExpertType, SpecializedExpert
        self.assertTrue(hasattr(BaseExpert, 'forward'))
        self.assertIn('REASONING', ExpertType.__members__)
        print("[OK] experts package")

    def test_import_routing(self):
        """Routing package should import cleanly."""
        from optimization_core.modules.feed_forward.routing import BaseRouter, RoutingResult, TokenLevelRouter
        self.assertTrue(hasattr(TokenLevelRouter, 'forward'))
        print("[OK] routing package")

    def test_import_layers(self):
        """Layers package should import cleanly."""
        from optimization_core.modules.feed_forward.layers import (
            FeedForwardBase, FeedForward, GatedFeedForward, SwiGLU, ReGLU, GeGLU, create_feed_forward
        )
        self.assertTrue(callable(create_feed_forward))
        print("[OK] layers package")

    def test_import_blocks(self):
        """Blocks package should import cleanly."""
        from optimization_core.modules.feed_forward.blocks import MoEBlock
        self.assertTrue(hasattr(MoEBlock, 'forward'))
        print("[OK] blocks package")

    def test_lazy_imports_from_main_init(self):
        """Lazy imports from feed_forward.__init__ should resolve."""
        from optimization_core.modules.feed_forward import (
            FeedForward, SwiGLU, BaseExpert, ExpertType, MoEBlock, TokenLevelRouter,
            BaseRouter, RoutingResult, SpecializedExpert
        )
        self.assertIsNotNone(FeedForward)
        self.assertIsNotNone(MoEBlock)
        print("[OK] lazy imports from feed_forward")

    def test_moe_forward_pass(self):
        """MoE block should run a forward pass."""
        import torch
        import torch.nn as nn
        from optimization_core.modules.feed_forward.experts.base import ExpertType
        from optimization_core.modules.feed_forward.experts.specialized import SpecializedExpert
        from optimization_core.modules.feed_forward.routing.token_router import TokenLevelRouter
        from optimization_core.modules.feed_forward.blocks.moe_block import MoEBlock

        hidden_size = 64
        expert_types = [ExpertType.REASONING, ExpertType.COMPUTATION]
        experts = nn.ModuleList([
            SpecializedExpert(hidden_size, et) for et in expert_types
        ])
        router = TokenLevelRouter(hidden_size, len(expert_types), expert_types, top_k=1)
        block = MoEBlock(router, experts)
        block.eval()

        x = torch.randn(1, 4, hidden_size)
        with torch.no_grad():
            out = block(x)
        
        self.assertEqual(out.shape, x.shape)
        print("[OK] MoE forward pass")

    def test_ffn_factory(self):
        """FFN factory should create all variants."""
        import torch
        from optimization_core.modules.feed_forward.layers.ffn import create_feed_forward

        for ff_type in ["standard", "gated", "swiglu", "reglu", "geglu"]:
            ffn = create_feed_forward(d_model=32, d_ff=64, ff_type=ff_type)
            x = torch.randn(1, 2, 32)
            out = ffn(x)
            self.assertEqual(out.shape, x.shape)
        print("[OK] FFN factory all variants")


if __name__ == "__main__":
    unittest.main(verbosity=2)

