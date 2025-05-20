| Model               | Description                                                                                          | GitHub Repository                                                                 |
|---------------------|------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| Reason-RFT          | Reinforcement Fine-Tuning framework enhancing generalization in visual reasoning tasks.              | [tanhuajie/Reason-RFT](https://github.com/tanhuajie/Reason-RFT)                   |
| R1-AQA              | Applies RL to audio question-answering tasks, outperforming supervised fine-tuning.                  | [xiaomi-research/r1-aqa](https://github.com/xiaomi-research/r1-aqa)               |
| UI-R1               | Enhances GUI agent action prediction using DeepSeek's R1-style rule-based RL.                        | [ritzz-ai/GUI-R1](https://github.com/ritzz-ai/GUI-R1)                              |
| Video-R1            | Explores the R1 paradigm for video reasoning within Multimodal Large Language Models (MLLMs).        | [tulerfeng/Video-R1](https://github.com/tulerfeng/Video-R1)                       |
| MetaSpatial         | Enhances 3D spatial reasoning in Vision-Language Models (VLMs) using RL for real-time scene generation. | [PzySeere/MetaSpatial](https://github.com/PzySeere/MetaSpatial)                   |
| OpenVLThinker-7B    | Fine-tuned version of Qwen2.5-7B-Instruct on the OpenThoughts-114k dataset.                          | [open-thoughts/OpenThinker-7B](https://huggingface.co/open-thoughts/OpenThinker-7B)|
| OThink-MR1          | Stimulates multimodal generalized reasoning capabilities via dynamic reinforcement learning.         | [modelscope/awesome-deep-reasoning](https://github.com/modelscope/awesome-deep-reasoning) |
| Skywork R1V         | Open-sourced multimodal reasoning model with advanced visual chain-of-thought capabilities.          | [SkyworkAI/Skywork-R1V](https://github.com/SkyworkAI/Skywork-R1V)                 |
| R1-VL               | Introduces StepGRPO for step-by-step reasoning in MLLMs.                                             | [jingyi0000/R1-VL](https://github.com/jingyi0000/R1-VL)                            |
| R1-Onevision        | Versatile multimodal reasoning large model integrating visual and textual data.                      | [Fancy-MLLM/R1-Onevision](https://github.com/Fancy-MLLM/R1-Onevision)             |
| VisualPRM           | Advanced multimodal Process Reward Model improving reasoning abilities in MLLMs.                     | [OpenGVLab/InternVL](https://github.com/OpenGVLab/InternVL)                        |
| LMM-R1              | High-performance RL infrastructure for enhancing multimodal reasoning capabilities.                   | [TideDra/lmm-r1](https://github.com/TideDra/lmm-r1)                                |
| Curr-ReFT           | Curriculum Reinforcement Fine-Tuning strategy to enhance out-of-distribution generalization and reasoning abilities. | [ding523/Curr_REFT](https://github.com/ding523/Curr_REFT)                          |
| VisRL               | Applies RL to intention-driven visual perception, optimizing visual reasoning without annotated intermediate bounding boxes. | [zhangquanchen/VisRL](https://github.com/zhangquanchen/VisRL)                     |
| MM-Eurek            | Extends large-scale rule-based RL to multimodal reasoning, introducing models like MM-Eureka-Qwen-7B. | [ModalMinds/MM-EUREKA](https://github.com/ModalMinds/MM-EUREKA)                   |

# Suvey 
https://arxiv.org/pdf/2504.21277

# Awesome 
https://github.com/yuanpinz/awesome-deep-multimodal-reasoning

| Method                                                      | Description                                                                                                                                                             | Impact on Reasoning Models                                                                                                                               | Example Models                                                              | Link                                            | GitHub Open Source                                         |
|-------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|-------------------------------------------------|------------------------------------------------------------|
| **Group Relative Policy Optimization (GRPO)**               | Eliminates the need for a traditional critic by employing group sampling to estimate advantages. It utilizes conservative policy updates to maintain stability.         | Significantly improves training efficiency and reduces memory consumption. It has shown particular benefits in enhancing mathematical reasoning abilities. | DeepSeek-Math, DeepSeek-R1                                                  | [GRPO Paper/Source](https://arxiv.org/abs/2402.02903)          | [GRPO-Zero](https://github.com/policy-gradient/GRPO-Zero) |
| **Reinforcement Learning with Verifiable Rewards (RLVR)**   | Leverages binary feedback (e.g., correct/incorrect) and does not require a separate learned reward model. It often relies on external tools like calculators or verifiers to provide feedback. | Enhances training efficiency, especially in domains where task outcomes are easily verifiable (e.g., math problems, code generation with unit tests).       | DeepSeek-R1                                                                 | [RLVR Paper/Source](https://arxiv.org/abs/2305.14340)             | [GSM8K-RLVR](https://github.com/Mohammadjafari80/GSM8K-RLVR) |
| **Dr. GRPO and Length-Controlled Policy Optimization (LCPO)** | These methods are extensions or related techniques designed to address specific challenges like length bias in model generations. They often incorporate mechanisms to penalize overly long and incorrect answers, thereby exerting more control over the length and quality of responses. | Leads to improved accuracy by discouraging verbose, unhelpful, or incorrect reasoning chains. Helps in reducing biases related to output length.             | Kimi k1.5, various other Large Language Models (LLMs) employing advanced RLHF | [LCPO Paper](https://arxiv.org/abs/2402.04831) | [Länge](https://github.com/cmu-l3/Länge)                   |

# GRPO: Mathematical Formulation

**Group Relative Policy Optimization (GRPO)** eliminates the need for a traditional critic by employing group sampling to estimate advantages and uses conservative policy updates for stability.

## Objective Function

The objective function for GRPO is:

```
J_GRPO(θ) = E_{q~P(Q), {o_i}~π_{θ_old}} [
    (1/G) ∑_{i=1}^G ∑_{t=1}^{|o_i|} min(r_t(θ)A_i, clip(r_t(θ), 1-ε, 1+ε)A_i)
    - β D_KL[π_θ || π_ref]
]
```

where

```
r_t(θ) = π_θ(o_{i,t} | q, o_{i,<t}) / π_{θ_old}(o_{i,t} | q, o_{i,<t})
```

**Definitions:**
- `q ~ P(Q)`: question sampled from distribution P(Q)
- `{o_i} ~ π_{θ_old}`: outputs sampled from the old policy
- `G`: number of samples
- `A_i`: advantage estimate for sample i
- `ε`: clipping parameter
- `β`: KL penalty coefficient
- `D_KL[π_θ || π_ref]`: KL divergence between current and reference policy

---

*This file documents the mathematical formulation for the GRPO algorithm. For implementation, see the corresponding Python files.*

## Production Code 

https://github.com/LLaVA-VL/LLaVA-NeXT
