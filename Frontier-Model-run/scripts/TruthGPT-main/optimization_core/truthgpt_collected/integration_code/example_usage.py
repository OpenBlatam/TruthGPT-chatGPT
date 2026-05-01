#!/usr/bin/env python3
"""
Ejemplo de uso de TruthGPT Advanced Integration
===============================================

Este script demuestra cómo usar todas las funcionalidades integradas
de TruthGPT Advanced.
"""

import torch
import numpy as np
from truthgpt_advanced_integration import (
    TruthGPTAdvanced,
    TruthGPTAdvancedConfig,
    MemoryConfig,
    RedundancySuppressionConfig,
    RLHFConfig,
    train_truthgpt_advanced
)


def example_1_basic_usage():
    """Ejemplo 1: Uso básico de TruthGPT Advanced."""
    print("\n" + "="*60)
    print("EJEMPLO 1: Uso Básico")
    print("="*60)
    
    # Crear configuración
    config = TruthGPTAdvancedConfig(
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        use_bulk_processing=True,
        enable_autonomous_agents=True,
        enable_memory_system=True
    )
    
    # Crear modelo
    model = TruthGPTAdvanced(config)
    model.eval()
    
    # Crear datos de ejemplo
    batch_size, seq_len = 4, 32
    inputs = torch.randn(batch_size, seq_len, config.hidden_dim)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(inputs, use_memory=True, suppress_redundancy=True)
    
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {outputs['output'].shape}")
    print(f"Memory used: {outputs['memory_used']}")
    print(f"Redundancy suppressed: {outputs['redundancy_suppressed']}")
    print(f"Hierarchical levels: {len(outputs['hierarchical_outputs'])}")


def example_2_memory_system():
    """Ejemplo 2: Sistema de memoria."""
    print("\n" + "="*60)
    print("EJEMPLO 2: Sistema de Memoria")
    print("="*60)
    
    config = TruthGPTAdvancedConfig(
        memory_config=MemoryConfig(
            memory_dim=256,
            max_memory_size=1000,
            retrieval_k=5
        )
    )
    
    model = TruthGPTAdvanced(config)
    model.eval()
    
    # Almacenar información en memoria
    print("Almacenando información en memoria...")
    for i in range(10):
        key = torch.randn(256)
        value = torch.randn(256)
        metadata = {'id': i, 'type': 'example'}
        model.store_in_memory(key, value, metadata)
    
    # Recuperar de memoria
    query = torch.randn(256)
    retrieved_values, retrieved_weights = model.memory_system.retrieve(query, k=5)
    
    print(f"Query shape: {query.shape}")
    print(f"Retrieved values shape: {retrieved_values.shape}")
    print(f"Retrieved weights shape: {retrieved_weights.shape}")
    print(f"Number of items in short-term memory: {len(model.memory_system.short_term_memory)}")


def example_3_redundancy_suppression():
    """Ejemplo 3: Supresión de redundancia."""
    print("\n" + "="*60)
    print("EJEMPLO 3: Supresión de Redundancia")
    print("="*60)
    
    config = TruthGPTAdvancedConfig(
        redundancy_config=RedundancySuppressionConfig(
            similarity_threshold=0.8,
            use_hierarchical_clustering=True,
            redundancy_detection_method="cosine"
        )
    )
    
    model = TruthGPTAdvanced(config)
    
    # Crear datos con redundancias
    base_item = torch.randn(32, 512)
    items = [base_item + 0.1 * torch.randn(32, 512) for _ in range(10)]  # Items similares
    items += [torch.randn(32, 512) for _ in range(5)]  # Items únicos
    
    print(f"Items originales: {len(items)}")
    
    # Procesar con supresión de redundancia
    unique_items = model.redundancy_suppressor.process_bulk(items, items)
    
    print(f"Items únicos después de supresión: {len(unique_items)}")
    print(f"Reducción: {len(items) - len(unique_items)} items eliminados")


def example_4_autonomous_agent():
    """Ejemplo 4: Agente autónomo con RLHF."""
    print("\n" + "="*60)
    print("EJEMPLO 4: Agente Autónomo RLHF")
    print("="*60)
    
    config = TruthGPTAdvancedConfig(
        rlhf_config=RLHFConfig(
            learning_rate=1e-4,
            discount_factor=0.99,
            exploration_rate=0.2
        ),
        enable_autonomous_agents=True
    )
    
    model = TruthGPTAdvanced(config)
    model.autonomous_agent.train()
    
    # Simular episodio de entrenamiento
    state_dim = config.hidden_dim
    num_steps = 20
    
    states = []
    actions = []
    rewards = []
    human_feedback = []
    
    print("Simulando episodio de entrenamiento...")
    for step in range(num_steps):
        # Estado aleatorio
        state = torch.randn(state_dim)
        states.append(state)
        
        # Seleccionar acción
        action, log_prob = model.autonomous_agent.select_action(state, training=True)
        actions.append(action)
        
        # Reward simulado (basado en acción)
        reward = 0.5 + 0.1 * np.sin(step * 0.1) + np.random.normal(0, 0.1)
        rewards.append(reward)
        
        # Human feedback simulado
        hf = 0.3 + 0.05 * np.cos(step * 0.1) + np.random.normal(0, 0.05)
        human_feedback.append(hf)
    
    # Entrenar agente
    training_stats = model.train_autonomous_agent(
        states, actions, rewards, human_feedback
    )
    
    print(f"Training completado:")
    print(f"  Policy Loss: {training_stats['policy_loss']:.4f}")
    print(f"  Value Loss: {training_stats['value_loss']:.4f}")
    print(f"  Total Loss: {training_stats['total_loss']:.4f}")
    print(f"  Mean Reward: {training_stats['mean_reward']:.4f}")


def example_5_training():
    """Ejemplo 5: Entrenamiento completo."""
    print("\n" + "="*60)
    print("EJEMPLO 5: Entrenamiento Completo")
    print("="*60)
    
    config = TruthGPTAdvancedConfig(
        hidden_dim=256,
        num_layers=3,
        num_heads=4,
        use_bulk_processing=True,
        enable_memory_system=True
    )
    
    model = TruthGPTAdvanced(config)
    
    # Crear datos de entrenamiento
    print("Generando datos de entrenamiento...")
    num_samples = 100
    seq_len = 16
    train_data = [
        torch.randn(seq_len, config.hidden_dim) 
        for _ in range(num_samples)
    ]
    
    print(f"Datos de entrenamiento: {len(train_data)} muestras")
    print(f"Shape por muestra: {train_data[0].shape}")
    
    # Entrenar (solo 2 épocas para ejemplo)
    print("\nIniciando entrenamiento...")
    trained_model = train_truthgpt_advanced(
        model=model,
        train_data=train_data,
        epochs=2,
        batch_size=8
    )
    
    print("\n✅ Entrenamiento completado!")


def example_6_full_pipeline():
    """Ejemplo 6: Pipeline completo."""
    print("\n" + "="*60)
    print("EJEMPLO 6: Pipeline Completo")
    print("="*60)
    
    # Configuración completa
    config = TruthGPTAdvancedConfig(
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        use_bulk_processing=True,
        enable_autonomous_agents=True,
        enable_memory_system=True,
        memory_config=MemoryConfig(
            memory_dim=512,
            max_memory_size=5000,
            retrieval_k=10
        ),
        redundancy_config=RedundancySuppressionConfig(
            similarity_threshold=0.85,
            redundancy_detection_method="cosine"
        ),
        rlhf_config=RLHFConfig(
            learning_rate=1e-4,
            exploration_rate=0.1
        )
    )
    
    model = TruthGPTAdvanced(config)
    model.eval()
    
    # 1. Procesar batch de inputs
    print("1. Procesando batch de inputs...")
    batch_size, seq_len = 8, 64
    inputs = torch.randn(batch_size, seq_len, config.hidden_dim)
    
    with torch.no_grad():
        outputs = model(inputs, use_memory=True, suppress_redundancy=True)
    
    print(f"   Output shape: {outputs['output'].shape}")
    
    # 2. Almacenar información importante en memoria
    print("\n2. Almacenando información en memoria...")
    for i in range(5):
        key = outputs['output'][i, -1, :]  # Último token de cada secuencia
        value = outputs['output'][i, -1, :]
        model.store_in_memory(key, value, {'sequence_id': i})
    
    print(f"   Items en memoria: {len(model.memory_system.short_term_memory)}")
    
    # 3. Procesar nuevo batch con memoria
    print("\n3. Procesando nuevo batch con memoria...")
    new_inputs = torch.randn(batch_size, seq_len, config.hidden_dim)
    with torch.no_grad():
        new_outputs = model(new_inputs, use_memory=True, suppress_redundancy=True)
    
    print(f"   Nuevo output shape: {new_outputs['output'].shape}")
    print(f"   Memoria utilizada: {new_outputs['memory_used']}")
    
    # 4. Entrenar agente autónomo
    print("\n4. Entrenando agente autónomo...")
    model.autonomous_agent.train()
    
    states = [torch.randn(config.hidden_dim) for _ in range(10)]
    actions = [i % config.hidden_dim for i in range(10)]
    rewards = [0.5 + 0.1 * i for i in range(10)]
    human_feedback = [0.3 + 0.05 * i for i in range(10)]
    
    training_stats = model.train_autonomous_agent(states, actions, rewards, human_feedback)
    print(f"   Training stats: {training_stats}")
    
    print("\n✅ Pipeline completo ejecutado exitosamente!")


def main():
    """Ejecutar todos los ejemplos."""
    print("="*60)
    print("TRUTHGPT ADVANCED INTEGRATION - EJEMPLOS DE USO")
    print("="*60)
    
    try:
        example_1_basic_usage()
        example_2_memory_system()
        example_3_redundancy_suppression()
        example_4_autonomous_agent()
        example_5_training()
        example_6_full_pipeline()
        
        print("\n" + "="*60)
        print("✅ TODOS LOS EJEMPLOS COMPLETADOS EXITOSAMENTE!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error ejecutando ejemplos: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()




