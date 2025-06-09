#!/usr/bin/env python3
import os
import yaml
import tyro
from kf_grpo_train import KFGRPOScriptArguments, main

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def config_to_args(config):
    """Convert YAML config to script arguments."""
    args_dict = {}
    
    # Basic dataset and output settings
    args_dict.update({
        'dataset_name': config['dataset_name'],
        'dataset_config': config['dataset_config'],
        'dataset_train_split': config['dataset_train_split'],
        'dataset_test_split': config['dataset_test_split'],
        'output_dir': config['output_dir']
    })
    
    # Model settings
    args_dict.update({
        'model_name': config['model']['name'],
        'use_deepspeed': config['model']['use_deepspeed'],
        'fp16': config['model']['fp16'],
        'bf16': config['model']['bf16']
    })
    
    # Training settings
    args_dict.update({
        'train_batch_size': config['training']['batch_size'],
        'gradient_accumulation_steps': config['training']['gradient_accumulation_steps'],
        'learning_rate': config['training']['learning_rate'],
        'max_steps': config['training']['max_steps'],
        'warmup_ratio': config['training']['warmup_ratio'],
        'weight_decay': config['training']['weight_decay'],
        'max_grad_norm': config['training']['max_grad_norm'],
        'lr_scheduler_type': config['training']['lr_scheduler_type'],
        'num_cycles': config['training']['num_cycles']
    })
    
    # Optimization settings
    args_dict.update({
        'use_amp': config['optimization']['use_amp'],
        'use_gradient_checkpointing': config['optimization']['use_gradient_checkpointing'],
        'use_flash_attention': config['optimization']['use_flash_attention'],
        'use_8bit_optimizer': config['optimization']['use_8bit_optimizer'],
        'use_cpu_offload': config['optimization']['use_cpu_offload'],
        'use_activation_checkpointing': config['optimization']['use_activation_checkpointing'],
        'use_attention_slicing': config['optimization']['use_attention_slicing'],
        'use_sequence_parallelism': config['optimization']['use_sequence_parallelism']
    })
    
    # Performance settings
    args_dict.update({
        'use_cudnn_benchmark': config['performance']['use_cudnn_benchmark'],
        'use_tf32': config['performance']['use_tf32'],
        'use_channels_last': config['performance']['use_channels_last'],
        'use_compile': config['performance']['use_compile']
    })
    
    # DeepSpeed settings
    args_dict.update({
        'deepspeed_config': {
            'zero_stage': config['deepspeed']['zero_stage'],
            'offload_optimizer': config['deepspeed']['offload_optimizer'],
            'offload_param': config['deepspeed']['offload_param'],
            'gradient_clipping': config['deepspeed']['gradient_clipping']
        } if config['deepspeed']['use_deepspeed'] else None
    })
    
    # DeepSeek V3 settings
    deepseek_config = {
        'model_type': config['deepseek']['model_type'],
        'use_native_implementation': config['deepseek'].get('use_native_implementation', True),
        'max_position_embeddings': config['deepseek']['max_position_embeddings'],
        'hidden_size': config['deepseek']['hidden_size'],
        'num_hidden_layers': config['deepseek']['num_hidden_layers'],
        'num_attention_heads': config['deepseek']['num_attention_heads'],
        'num_key_value_heads': config['deepseek'].get('num_key_value_heads'),
        'vocab_size': config['deepseek'].get('vocab_size', 102400),
        'intermediate_size': config['deepseek']['intermediate_size'],
        'hidden_dropout_prob': config['deepseek']['hidden_dropout_prob'],
        'attention_dropout_prob': config['deepseek']['attention_dropout_prob'],
        'layer_norm_eps': config['deepseek']['layer_norm_eps'],
        'rope_theta': config['deepseek'].get('rope_theta', 10000.0),
        'q_lora_rank': config['deepseek'].get('q_lora_rank', 1536),
        'kv_lora_rank': config['deepseek'].get('kv_lora_rank', 512),
        'qk_rope_head_dim': config['deepseek'].get('qk_rope_head_dim', 64),
        'v_head_dim': config['deepseek'].get('v_head_dim', 128),
        'qk_nope_head_dim': config['deepseek'].get('qk_nope_head_dim', 128),
        'n_routed_experts': config['deepseek'].get('n_routed_experts', 64),
        'n_shared_experts': config['deepseek'].get('n_shared_experts', 2),
        'n_activated_experts': config['deepseek'].get('n_activated_experts', 6),
        'moe_intermediate_size': config['deepseek'].get('moe_intermediate_size', 1407),
        'shared_intermediate_size': config['deepseek'].get('shared_intermediate_size', 1024),
        'use_fp8': config['deepseek'].get('use_fp8', False),
        'original_seq_len': config['deepseek'].get('original_seq_len', 4096),
        'rope_factor': config['deepseek'].get('rope_factor', 40),
        'beta_fast': config['deepseek'].get('beta_fast', 32),
        'beta_slow': config['deepseek'].get('beta_slow', 1),
        'mscale': config['deepseek'].get('mscale', 1.0),
        'use_rotary_embeddings': config['deepseek']['use_rotary_embeddings'],
        'use_alibi': config['deepseek']['use_alibi'],
        'use_flash_attention_2': config['deepseek']['use_flash_attention_2'],
        'use_sliding_window': config['deepseek']['use_sliding_window'],
        'sliding_window_size': config['deepseek']['sliding_window_size']
    }
    
    args_dict.update({
        'deepseek_config': deepseek_config
    })
    
    # Parallel processing settings
    args_dict.update({
        'parallel_config': config['parallel']
    })
    
    # Kalman filter settings
    args_dict.update({
        'kalman_config': {
            'process_noise': config['kalman']['process_noise'],
            'measurement_noise': config['kalman']['measurement_noise'],
            'memory_size': config['kalman']['memory_size']
        }
    })
    
    # Reward functions
    args_dict.update({
        'reward_funcs': config['reward_funcs']
    })
    
    # Distributed settings
    args_dict.update({
        'distributed_config': {
            'backend': config['distributed']['backend'],
            'world_size': config['distributed']['world_size'],
            'rank': config['distributed']['rank'],
            'master_addr': config['distributed']['master_addr'],
            'master_port': config['distributed']['master_port']
        }
    })
    
    return KFGRPOScriptArguments(**args_dict)

def main_with_config():
    """Main function to run training with YAML config."""
    import argparse
    parser = argparse.ArgumentParser(description='Run KF-GRPO training with YAML config')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()
    
    # Load and convert config
    config = load_config(args.config)
    script_args = config_to_args(config)
    
    # Run training
    main(script_args)

if __name__ == "__main__":
    main_with_config()  