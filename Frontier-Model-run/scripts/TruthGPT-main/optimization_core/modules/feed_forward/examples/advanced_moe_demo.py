"""
Advanced MoE Demo
=================

Demonstrates the unified Feed Forward architecture:
- Creating specialized experts
- Configuring a token-level router
- Assembling an MoE block
- Running a forward pass
"""

import sys
import os

# Ensure the package root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import torch
import torch.nn as nn

from optimization_core.modules.feed_forward.experts.base import ExpertType
from optimization_core.modules.feed_forward.experts.specialized import SpecializedExpert
from optimization_core.modules.feed_forward.routing.token_router import TokenLevelRouter
from optimization_core.modules.feed_forward.blocks.moe_block import MoEBlock
from optimization_core.modules.feed_forward.layers.ffn import create_feed_forward


def demo_unified_moe():
    """Demonstrate the unified MoE architecture."""
    print("=" * 70)
    print("  TruthGPT - Unified Feed Forward Architecture Demo")
    print("=" * 70)

    hidden_size = 256
    num_experts = 4
    batch_size = 2
    seq_len = 8

    # 1. Define expert types
    expert_types = [
        ExpertType.REASONING,
        ExpertType.COMPUTATION,
        ExpertType.MATHEMATICAL,
        ExpertType.LOGICAL,
    ]
    print(f"\n[1] Expert Types: {[et.value for et in expert_types]}")

    # 2. Create specialized experts
    experts = nn.ModuleList([
        SpecializedExpert(
            hidden_size=hidden_size,
            expert_type=etype,
            dropout=0.1
        )
        for etype in expert_types
    ])
    
    total_params = sum(p.numel() for e in experts for p in e.parameters())
    print(f"[2] Created {len(experts)} experts ({total_params:,} params)")
    for i, expert in enumerate(experts):
        info = expert.get_info()
        print(f"    Expert {i}: type={info['type']}, params={info['params']:,}")

    # 3. Create token-level router
    router = TokenLevelRouter(
        hidden_size=hidden_size,
        num_experts=num_experts,
        expert_types=expert_types,
        top_k=1,
        temperature=1.0,
        load_balance_weight=0.1,
        use_gating=True,
        use_auxiliary_loss=True,
        dropout=0.1
    )
    router_params = sum(p.numel() for p in router.parameters())
    print(f"[3] Router created ({router_params:,} params)")

    # 4. Assemble MoE Block
    moe_block = MoEBlock(
        router=router,
        experts=experts,
        use_residual=True
    )
    block_params = sum(p.numel() for p in moe_block.parameters())
    print(f"[4] MoE Block assembled ({block_params:,} total params)")

    # 5. Forward pass
    x = torch.randn(batch_size, seq_len, hidden_size)
    print(f"\n[5] Input shape: {x.shape}")

    moe_block.eval()
    with torch.no_grad():
        output = moe_block(x)
    
    print(f"    Output shape: {output.shape}")
    print(f"    Output mean: {output.mean():.6f}")
    print(f"    Output std:  {output.std():.6f}")

    # 6. Forward pass with routing info
    with torch.no_grad():
        output_info, metadata = moe_block(x, return_routing_info=True)
    
    aux_loss = metadata.get("auxiliary_loss", 0.0)
    print(f"\n[6] Routing metadata:")
    print(f"    Auxiliary loss: {aux_loss:.6f}")

    # 7. Demo standard FFN layers
    print(f"\n[7] FFN Layer Variants:")
    for ff_type in ["standard", "swiglu", "reglu", "geglu", "gated"]:
        ffn = create_feed_forward(d_model=hidden_size, d_ff=hidden_size * 4, ff_type=ff_type)
        test_input = torch.randn(1, 4, hidden_size)
        test_output = ffn(test_input)
        params = sum(p.numel() for p in ffn.parameters())
        print(f"    {ff_type:10s} -> output={test_output.shape}, params={params:,}")

    print("\n" + "=" * 70)
    print("  ✅ Demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    demo_unified_moe()

