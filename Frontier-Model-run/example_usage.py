#!/usr/bin/env python3
"""
Example Usage of Frontier Model Variants
Shows how to use different model architectures in practice.
"""

import os
import sys
import torch
import torch.nn.functional as F
from typing import Dict, Any, List

# Add variants to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'variants'))

def example_native_transformer():
    """Example usage of Native Transformer."""
    print("=== Native Transformer Example ===")
    
    from native_transformer.model import NativeTransformerForCausalLM, NativeTransformerConfig
    
    # Create config
    config = NativeTransformerConfig(
        vocab_size=1000,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        use_adaptive_attention=True,
        use_sparse_attention=True,
        use_rotary_embeddings=True,
    )
    
    # Create model
    model = NativeTransformerForCausalLM(config)
    model.eval()
    
    # Example input
    input_ids = torch.randint(0, 1000, (2, 50))  # batch_size=2, seq_len=50
    attention_mask = torch.ones_like(input_ids)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs['logits']
        
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {logits.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Generate next token probabilities
        next_token_logits = logits[:, -1, :]  # Last token
        next_token_probs = F.softmax(next_token_logits, dim=-1)
        next_tokens = torch.multinomial(next_token_probs, num_samples=1)
        print(f"Next tokens: {next_tokens.squeeze().tolist()}")


def example_mixture_of_experts():
    """Example usage of Mixture of Experts."""
    print("\n=== Mixture of Experts Example ===")
    
    from mixture_of_experts.model import MoETransformerForCausalLM, MoETransformerConfig
    
    # Create config
    config = MoETransformerConfig(
        vocab_size=1000,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_experts=8,
        num_experts_per_token=2,
        load_balancing_loss_coef=0.01,
    )
    
    # Create model
    model = MoETransformerForCausalLM(config)
    model.eval()
    
    # Example input
    input_ids = torch.randint(0, 1000, (2, 50))
    attention_mask = torch.ones_like(input_ids)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs['logits']
        aux_losses = outputs.get('aux_losses', [])
        
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {logits.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Auxiliary losses: {len(aux_losses)} layers")
        
        # Show expert usage
        if aux_losses:
            print("Expert routing information available")


def example_retrieval_augmented():
    """Example usage of Retrieval Augmented Generation."""
    print("\n=== Retrieval Augmented Generation Example ===")
    
    from retrieval_augmented.model import RAGTransformerForCausalLM, RAGTransformerConfig
    
    # Create config
    config = RAGTransformerConfig(
        vocab_size=1000,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        retrieval_dim=256,
        num_retrieved_docs=5,
        use_cross_attention=True,
        use_relevance_scoring=True,
    )
    
    # Create model
    model = RAGTransformerForCausalLM(config)
    model.eval()
    
    # Example input
    input_ids = torch.randint(0, 1000, (2, 50))
    attention_mask = torch.ones_like(input_ids)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs['logits']
        retrieval_info = outputs.get('retrieval_info', [])
        
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {logits.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Retrieval info: {len(retrieval_info)} layers")
        
        # Show retrieval information
        if retrieval_info:
            for i, info in enumerate(retrieval_info):
                if info:  # Check if layer has retrieval
                    print(f"Layer {i}: Retrieved documents available")


def example_multi_modal():
    """Example usage of Multi-Modal Transformer."""
    print("\n=== Multi-Modal Transformer Example ===")
    
    from multi_modal.model import MultiModalTransformerForCausalLM, MultiModalTransformerConfig
    
    # Create config
    config = MultiModalTransformerConfig(
        vocab_size=1000,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        modalities=["text", "vision"],
        use_cross_modal_attention=True,
        fusion_method="attention",
    )
    
    # Create model
    model = MultiModalTransformerForCausalLM(config)
    model.eval()
    
    # Example input
    input_ids = torch.randint(0, 1000, (2, 50))
    attention_mask = torch.ones_like(input_ids)
    images = torch.randn(2, 3, 224, 224)  # Batch of images
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            images=images
        )
        logits = outputs['logits']
        cross_modal_info = outputs.get('cross_modal_info', [])
        vision_features = outputs.get('vision_features')
        
        print(f"Text input shape: {input_ids.shape}")
        print(f"Image input shape: {images.shape}")
        print(f"Output shape: {logits.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        if vision_features is not None:
            print(f"Vision features shape: {vision_features.shape}")
        
        # Show cross-modal fusion
        fused_layers = sum(1 for info in cross_modal_info if info)
        print(f"Cross-modal fusion in {fused_layers} layers")


def example_reinforcement_learning():
    """Example usage of Reinforcement Learning Transformer."""
    print("\n=== Reinforcement Learning Transformer Example ===")
    
    from reinforcement_learning.model import RLTransformerForCausalLM, RLTransformerConfig
    
    # Create config
    config = RLTransformerConfig(
        vocab_size=1000,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        use_multi_objective=True,
        reward_types=["accuracy", "fluency"],
        use_experience_replay=True,
    )
    
    # Create model
    model = RLTransformerForCausalLM(config)
    model.eval()
    
    # Example input
    input_ids = torch.randint(0, 1000, (2, 50))
    attention_mask = torch.ones_like(input_ids)
    
    # Example rewards
    rewards = {
        "accuracy": torch.randn(2, 50),
        "fluency": torch.randn(2, 50)
    }
    
    # Forward pass with RL
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            rewards=rewards,
            compute_values=True
        )
        logits = outputs['logits']
        rl_info = outputs.get('rl_info', [])
        value_loss = outputs.get('value_loss')
        
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {logits.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"RL info: {len(rl_info)} layers")
        
        if value_loss is not None:
            print(f"Value loss: {value_loss.item():.4f}")
        
        # Show RL generation
        generated = model.generate_with_rl(
            input_ids=input_ids[:1],  # Single example
            max_length=20,
            temperature=0.8
        )
        print(f"Generated sequence length: {generated['generated_ids'].shape[1]}")


def example_training_setup():
    """Example of setting up training for any variant."""
    print("\n=== Training Setup Example ===")
    
    from native_transformer.model import NativeTransformerForCausalLM, NativeTransformerConfig
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create model
    config = NativeTransformerConfig(vocab_size=1000, hidden_size=256, num_hidden_layers=4)
    model = NativeTransformerForCausalLM(config)
    
    # Create dummy dataset
    num_samples = 100
    seq_length = 64
    input_ids = torch.randint(0, 1000, (num_samples, seq_length))
    labels = input_ids.clone()
    
    dataset = TensorDataset(input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    
    # Training loop example
    model.train()
    total_loss = 0
    
    print("Running training example...")
    for batch_idx, (batch_input_ids, batch_labels) in enumerate(dataloader):
        if batch_idx >= 3:  # Just a few batches for demo
            break
            
        optimizer.zero_grad()
        
        outputs = model(input_ids=batch_input_ids, labels=batch_labels)
        loss = outputs['loss']
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print(f"Batch {batch_idx + 1}: Loss = {loss.item():.4f}")
    
    avg_loss = total_loss / 3
    print(f"Average loss: {avg_loss:.4f}")


def main():
    """Run all examples."""
    print("Frontier Model Variants - Example Usage")
    print("=" * 50)
    
    try:
        example_native_transformer()
    except Exception as e:
        print(f"Native Transformer example failed: {e}")
    
    try:
        example_mixture_of_experts()
    except Exception as e:
        print(f"MoE example failed: {e}")
    
    try:
        example_retrieval_augmented()
    except Exception as e:
        print(f"RAG example failed: {e}")
    
    try:
        example_multi_modal()
    except Exception as e:
        print(f"Multi-modal example failed: {e}")
    
    try:
        example_reinforcement_learning()
    except Exception as e:
        print(f"RL example failed: {e}")
    
    try:
        example_training_setup()
    except Exception as e:
        print(f"Training example failed: {e}")
    
    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    main()