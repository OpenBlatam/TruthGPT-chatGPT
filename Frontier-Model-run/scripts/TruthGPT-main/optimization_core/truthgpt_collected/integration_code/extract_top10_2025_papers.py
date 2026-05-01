#!/usr/bin/env python3
"""
Script para extraer información de los Top 10 Papers de 2025
que han mejorado los benchmarks de LLMs y generar JSONs para integración.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import re

# Top 10 Papers de 2025 que mejoraron benchmarks
TOP_10_PAPERS_2025 = [
    {
        "paper_id": "qwen3_technical_report",
        "arxiv_id": "2505.09388",
        "title": "Qwen3 Technical Report",
        "authors": ["Alibaba Team"],
        "year": 2025,
        "url": "https://arxiv.org/abs/2505.09388",
        "contributions": [
            "SOTA en 14/15 benchmarks",
            "85.7% en AIME'24",
            "70.7% en LiveCodeBench v5",
            "Supera DeepSeek-V3 en multitarea"
        ],
        "key_techniques": [
            "Modos de pensamiento integrados",
            "Soporte multilingüe expandido (119 idiomas)",
            "Arquitectura multimodal",
            "Optimización de razonamiento"
        ],
        "benchmarks": [
            "AIME'24: 85.7%",
            "LiveCodeBench v5: 70.7%",
            "MMLU",
            "GSM8K",
            "AIME",
            "LiveCodeBench",
            "MMMU"
        ],
        "improvements": "+5-20% en precisión en múltiples benchmarks",
        "category": "multimodal_reasoning"
    },
    {
        "paper_id": "absolute_zero_azr",
        "arxiv_id": "2505.03335",
        "title": "Absolute Zero: Reinforced Self-play Reasoning with Zero Data",
        "authors": ["Autores no especificados"],
        "year": 2025,
        "url": "https://arxiv.org/abs/2505.03335",
        "contributions": [
            "SOTA en codificación y matemáticas",
            "AZR-Coder-7B: 50.4% promedio (+1.8 puntos vs. previos)",
            "Ganancias escalables (+13.2% en 14B)"
        ],
        "key_techniques": [
            "RLVR (Reinforcement Learning from Verifier Rewards)",
            "Self-play sin datos humanos",
            "Generación y resolución de tareas de razonamiento",
            "Paradigma de aprendizaje autónomo"
        ],
        "benchmarks": [
            "Codificación: +1.8 puntos",
            "Matemáticas: mejoras significativas",
            "Escalabilidad: +13.2% en 14B"
        ],
        "improvements": "+1.8 puntos en promedio, +13.2% escalable",
        "category": "reinforcement_learning"
    },
    {
        "paper_id": "seed1_5_vl",
        "arxiv_id": "2505.07062",
        "title": "Seed1.5-VL Technical Report",
        "authors": ["Autores no especificados"],
        "year": 2025,
        "url": "https://arxiv.org/abs/2505.07062",
        "contributions": [
            "SOTA en 38/60 benchmarks",
            "77.9% en MMMU con modo thinking",
            "Excelso en documentos y tareas agenticas"
        ],
        "key_techniques": [
            "Modelo multimodal compacto",
            "Comprensión y razonamiento general",
            "Modo thinking integrado",
            "Procesamiento de documentos"
        ],
        "benchmarks": [
            "MMMU: 77.9% (con modo thinking)",
            "38/60 benchmarks: SOTA",
            "Tareas agenticas: excelente rendimiento"
        ],
        "improvements": "SOTA en 38/60 benchmarks",
        "category": "multimodal"
    },
    {
        "paper_id": "mixture_of_reasonings",
        "arxiv_id": "2507.00606",
        "title": "Mixture of Reasonings: Teach Large Language Models to Reason with Adaptive Strategies",
        "authors": ["Autores no especificados"],
        "year": 2025,
        "url": "https://arxiv.org/abs/2507.00606",
        "contributions": [
            "Mejora razonamiento task-specific (+10-15% en benchmarks de multiturn)",
            "Elimina prompts frágiles",
            "Estrategias adaptativas"
        ],
        "key_techniques": [
            "Framework MoR (Mixture of Reasonings)",
            "Estrategias adaptativas",
            "Templates diversos",
            "Entrenamiento en múltiples estrategias"
        ],
        "benchmarks": [
            "Multiturn reasoning: +10-15%",
            "Task-specific reasoning: mejoras significativas"
        ],
        "improvements": "+10-15% en benchmarks de multiturn",
        "category": "reasoning"
    },
    {
        "paper_id": "crft_critical_representation",
        "arxiv_id": "2507.10085",
        "title": "Enhancing Chain-of-Thought Reasoning with Critical Representation Fine-tuning",
        "authors": ["Autores no especificados"],
        "year": 2025,
        "url": "https://arxiv.org/abs/2507.10085",
        "contributions": [
            "+16.4% en tareas de razonamiento one-shot",
            "Usa solo 0.016% de parámetros",
            "Supera PEFT tradicional"
        ],
        "key_techniques": [
            "CRFT (Critical Representation Fine-tuning)",
            "Fine-tuning ligero",
            "Enfoque en paths influyentes",
            "Optimización de representaciones críticas"
        ],
        "benchmarks": [
            "Razonamiento one-shot: +16.4%",
            "Eficiencia: 0.016% de parámetros"
        ],
        "improvements": "+16.4% en razonamiento one-shot",
        "category": "fine_tuning"
    },
    {
        "paper_id": "meta_cot_system2",
        "arxiv_id": "2501.04682",
        "title": "Towards System 2 Reasoning in LLMs: Learning How to Think With Meta Chain-of-Thought (Meta-CoT)",
        "authors": ["Autores no especificados"],
        "year": 2025,
        "url": "https://huggingface.co/papers/2501.04682",
        "contributions": [
            "Modelos RL superan instruction-tuned en problemas complejos",
            "+5-10% en razonamiento",
            "Enfatiza RL en 2025"
        ],
        "key_techniques": [
            "Framework para razonamiento iterativo y verificado",
            "MDPs (Markov Decision Processes)",
            "Meta-RL",
            "System 2 reasoning"
        ],
        "benchmarks": [
            "Razonamiento complejo: +5-10%",
            "Problemas complejos: mejoras significativas"
        ],
        "improvements": "+5-10% en razonamiento",
        "category": "reinforcement_learning"
    },
    {
        "paper_id": "sft_rl_generalization",
        "arxiv_id": "2501.17161",
        "title": "SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training",
        "authors": ["Autores no especificados"],
        "year": 2025,
        "url": "https://arxiv.org/abs/2501.17161",
        "contributions": [
            "RL mejora OOD y reconocimiento visual",
            "+8-12% en tareas de razonamiento textual/visual vs. SFT"
        ],
        "key_techniques": [
            "Estudio empírico comparando SFT y RL",
            "Generalización OOD",
            "Reconocimiento visual mejorado",
            "Post-training con RL"
        ],
        "benchmarks": [
            "Razonamiento textual/visual: +8-12% vs. SFT",
            "OOD: mejoras significativas",
            "Reconocimiento visual: mejoras"
        ],
        "improvements": "+8-12% en razonamiento textual/visual",
        "category": "training"
    },
    {
        "paper_id": "learning_dynamics_finetuning",
        "arxiv_id": "tPNHOoZFl9",
        "title": "Learning Dynamics of LLM Finetuning",
        "authors": ["Yi Ren", "Danica Sutherland"],
        "year": 2025,
        "url": "https://openreview.net/forum?id=tPNHOoZFl9",
        "contributions": [
            "Reduce alucinaciones",
            "Reduce 'squeezing effect'",
            "+5-10% en precisión de respuestas correctas en benchmarks de QA"
        ],
        "key_techniques": [
            "Trackear cambios en probabilidades durante fine-tuning",
            "Análisis de dinámicas de aprendizaje",
            "Detección de alucinaciones",
            "Mitigación de squeezing effect"
        ],
        "benchmarks": [
            "QA benchmarks: +5-10% en precisión",
            "Reducción de alucinaciones: significativa"
        ],
        "improvements": "+5-10% en precisión de QA",
        "category": "fine_tuning"
    },
    {
        "paper_id": "faster_cascades_speculative",
        "arxiv_id": "vo9t20wsmd",
        "title": "Faster Cascades via Speculative Decoding",
        "authors": ["Harikrishna Narasimhan", "et al."],
        "year": 2025,
        "url": "https://openreview.net/forum?id=vo9t20wsmd",
        "contributions": [
            "Acelera respuestas",
            "Reduce costos computacionales",
            "Mantiene calidad",
            "+15-20% en velocidad en benchmarks de inferencia"
        ],
        "key_techniques": [
            "Combina cascades y decoding especulativo",
            "Optimización de inferencia",
            "Reducción de latencia",
            "Eficiencia computacional"
        ],
        "benchmarks": [
            "Velocidad de inferencia: +15-20%",
            "Reducción de costos: significativa"
        ],
        "improvements": "+15-20% en velocidad de inferencia",
        "category": "inference"
    },
    {
        "paper_id": "deepseek_v3_insights",
        "arxiv_id": "2505.09343",
        "title": "Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures",
        "authors": ["DeepSeek Team"],
        "year": 2025,
        "url": "https://arxiv.org/abs/2505.09343",
        "contributions": [
            "Mejora eficiencia en memoria y cómputo",
            "MLA y MoE",
            "+10-15% en benchmarks de modelos masivos como GSM8K"
        ],
        "key_techniques": [
            "Co-diseño hardware-modelo",
            "MLA (Multi-head Latent Attention)",
            "MoE (Mixture of Experts)",
            "Escalado eficiente"
        ],
        "benchmarks": [
            "GSM8K: +10-15%",
            "Modelos masivos: mejoras significativas"
        ],
        "improvements": "+10-15% en benchmarks de modelos masivos",
        "category": "architecture"
    }
]


def extract_technical_details(paper_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrae detalles técnicos específicos de cada paper.
    """
    paper_id = paper_info["paper_id"]
    details = {
        "paper_id": paper_id,
        "arxiv_id": paper_info.get("arxiv_id", ""),
        "title": paper_info["title"],
        "authors": paper_info["authors"],
        "year": paper_info["year"],
        "url": paper_info["url"],
        "contributions": paper_info["contributions"],
        "key_techniques": paper_info["key_techniques"],
        "benchmarks": paper_info["benchmarks"],
        "improvements": paper_info["improvements"],
        "category": paper_info["category"],
        "technical_details": {}
    }
    
    # Detalles técnicos específicos por paper
    if paper_id == "qwen3_technical_report":
        details["technical_details"] = {
            "model_family": "Qwen3",
            "multilingual_support": 119,
            "thinking_modes": ["standard", "thinking", "multimodal"],
            "architecture": "transformer_with_thinking",
            "multimodal": True
        }
    
    elif paper_id == "absolute_zero_azr":
        details["technical_details"] = {
            "method": "RLVR",
            "self_play": True,
            "zero_data": True,
            "verifier_rewards": True,
            "model_sizes": ["7B", "14B"],
            "scalability": True
        }
    
    elif paper_id == "seed1_5_vl":
        details["technical_details"] = {
            "model_type": "multimodal",
            "compact": True,
            "thinking_mode": True,
            "document_processing": True,
            "agentic_tasks": True
        }
    
    elif paper_id == "mixture_of_reasonings":
        details["technical_details"] = {
            "framework": "MoR",
            "adaptive_strategies": True,
            "multiple_templates": True,
            "task_specific": True,
            "eliminates_fragile_prompts": True
        }
    
    elif paper_id == "crft_critical_representation":
        details["technical_details"] = {
            "method": "CRFT",
            "parameter_efficiency": 0.00016,  # 0.016%
            "critical_paths": True,
            "lightweight_finetuning": True,
            "one_shot_reasoning": True
        }
    
    elif paper_id == "meta_cot_system2":
        details["technical_details"] = {
            "framework": "Meta-CoT",
            "system2_reasoning": True,
            "mdp_based": True,
            "meta_rl": True,
            "iterative_verification": True
        }
    
    elif paper_id == "sft_rl_generalization":
        details["technical_details"] = {
            "comparison": "SFT vs RL",
            "ood_generalization": True,
            "visual_recognition": True,
            "post_training": True,
            "empirical_study": True
        }
    
    elif paper_id == "learning_dynamics_finetuning":
        details["technical_details"] = {
            "method": "Learning Dynamics Analysis",
            "probability_tracking": True,
            "hallucination_reduction": True,
            "squeezing_effect_mitigation": True,
            "qa_optimization": True
        }
    
    elif paper_id == "faster_cascades_speculative":
        details["technical_details"] = {
            "method": "Cascades + Speculative Decoding",
            "inference_optimization": True,
            "latency_reduction": True,
            "cost_reduction": True,
            "quality_preservation": True
        }
    
    elif paper_id == "deepseek_v3_insights":
        details["technical_details"] = {
            "architecture": "MLA + MoE",
            "hardware_co_design": True,
            "memory_efficiency": True,
            "computation_efficiency": True,
            "scaling": True
        }
    
    return details


def save_papers_json(output_dir: Path):
    """
    Guarda información de todos los papers en JSON.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_papers = {}
    
    for paper_info in TOP_10_PAPERS_2025:
        details = extract_technical_details(paper_info)
        paper_id = details["paper_id"]
        all_papers[paper_id] = details
        
        # Guardar individual
        output_file = output_dir / f"{paper_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(details, f, indent=2, ensure_ascii=False)
        print(f"✅ Guardado: {output_file}")
    
    # Guardar resumen
    summary_file = output_dir / "top10_2025_papers_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_papers, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Total papers: {len(all_papers)}")
    print(f"📁 Resumen guardado en: {summary_file}")
    
    return all_papers


def generate_integration_mapping():
    """
    Genera un mapeo de cómo integrar cada paper.
    """
    mapping = {
        "qwen3_technical_report": {
            "module_name": "Qwen3Module",
            "config_name": "Qwen3Config",
            "file_path": "papers/research/paper_qwen3.py",
            "integration_point": "multimodal_reasoning",
            "dependencies": []
        },
        "absolute_zero_azr": {
            "module_name": "AbsoluteZeroModule",
            "config_name": "AbsoluteZeroConfig",
            "file_path": "papers/research/paper_absolute_zero.py",
            "integration_point": "reinforcement_learning",
            "dependencies": []
        },
        "seed1_5_vl": {
            "module_name": "Seed1_5VLModule",
            "config_name": "Seed1_5VLConfig",
            "file_path": "papers/research/paper_seed1_5_vl.py",
            "integration_point": "multimodal",
            "dependencies": []
        },
        "mixture_of_reasonings": {
            "module_name": "MixtureOfReasoningsModule",
            "config_name": "MixtureOfReasoningsConfig",
            "file_path": "papers/research/paper_mixture_of_reasonings.py",
            "integration_point": "reasoning",
            "dependencies": []
        },
        "crft_critical_representation": {
            "module_name": "CRFTModule",
            "config_name": "CRFTConfig",
            "file_path": "papers/research/paper_crft.py",
            "integration_point": "fine_tuning",
            "dependencies": []
        },
        "meta_cot_system2": {
            "module_name": "MetaCoTModule",
            "config_name": "MetaCoTConfig",
            "file_path": "papers/research/paper_meta_cot.py",
            "integration_point": "reinforcement_learning",
            "dependencies": []
        },
        "sft_rl_generalization": {
            "module_name": "SFTRLGeneralizationModule",
            "config_name": "SFTRLGeneralizationConfig",
            "file_path": "papers/research/paper_sft_rl_generalization.py",
            "integration_point": "training",
            "dependencies": []
        },
        "learning_dynamics_finetuning": {
            "module_name": "LearningDynamicsModule",
            "config_name": "LearningDynamicsConfig",
            "file_path": "papers/research/paper_learning_dynamics.py",
            "integration_point": "fine_tuning",
            "dependencies": []
        },
        "faster_cascades_speculative": {
            "module_name": "FasterCascadesModule",
            "config_name": "FasterCascadesConfig",
            "file_path": "papers/inference/paper_faster_cascades.py",
            "integration_point": "inference",
            "dependencies": []
        },
        "deepseek_v3_insights": {
            "module_name": "DeepSeekV3Module",
            "config_name": "DeepSeekV3Config",
            "file_path": "papers/architecture/paper_deepseek_v3.py",
            "integration_point": "architecture",
            "dependencies": []
        }
    }
    
    return mapping


if __name__ == "__main__":
    # Directorio de salida
    output_dir = Path(__file__).parent / 'scraped_papers' / 'top10_2025'
    papers_data = save_papers_json(output_dir)
    
    # Generar mapeo de integración
    integration_mapping = generate_integration_mapping()
    mapping_file = output_dir / "integration_mapping.json"
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(integration_mapping, f, indent=2, ensure_ascii=False)
    print(f"📋 Mapeo de integración guardado en: {mapping_file}")
    
    print("\n📊 Papers procesados:")
    for key, data in papers_data.items():
        print(f"  - {key}: {data['title'][:60]}...")
        print(f"    📈 Mejoras: {data['improvements']}")
        print(f"    🏷️  Categoría: {data['category']}")
        print()



