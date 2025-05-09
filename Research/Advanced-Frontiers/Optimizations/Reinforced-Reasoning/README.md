

| Model           | Link                                                                                                                                                            | # Params            | RL Methods       | Fine-Tuning    | Architecture Type | Open | TTS |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | ---------------- | -------------- | ----------------- | ---- | --- |
| DeepSeek-V2     | [GitHub](https://github.com/deepseek-ai/DeepSeek-V2)                                                                                                            | 236B-A21B           | GRPO             | DPO + GRPO     | MoE               | ✓    | ✓   |
| GPT 4.5         | [GitHub Copilot](https://github.blog/changelog/2025-02-27-openai-gpt-4-5-in-github-copilot-now-available-in-public-preview/)                                    | -                   | RLHF, PPO, RBRM  | SFT + RLHF     | MoE               | ✗    | ✓   |
| Gemini          | [Support Page](https://support.google.com/gemini/answer/16176929?hl=en)                                                                                         | -                   | RLHF             | SFT + RLHF     | Single Model      | ✗    | ✗   |
| Claude 3.7      | [Anthropic](https://www.anthropic.com/news/claude-3-7-sonnet)                                                                                                   | -                   | RLAIF            | SFT + RLAIF    | Single Model      | ✗    | ✗   |
| Reka            | [GitHub](https://github.com/reka-ai)                                                                                                                            | 7B, 21B             | RLHF, PPO        | SFT + RLHF     | Single Model      | ✗    | ✗   |
| DeepSeekR1      | [GitHub](https://github.com/deepseek-ai/DeepSeek-R1)                                                                                                            | 240B-A22B           | GRPO             | DPO + GRPO     | MoE               | ✓    | ✓   |
| Nemotron-4 340B | [GitHub](https://github.com/NVIDIA/NeMo-Curator/blob/main/tutorials/nemotron_340B_synthetic_datagen/synthetic_preference_data_generation_nemotron_4_340B.ipynb) | 340B                | DPO, RPO         | DPO + RPO      | Single Model      | ✗    | ✗   |
| Falcon          | [GitHub](https://github.com/Decentralised-AI/falcon-40b)                                                                                                        | 40B                 | -                | SFT            | Single Model      | ✓    | ✗   |
| GPT-4           | [GitHub Topics](https://github.com/topics/gpt-4)                                                                                                                | -                   | RLHF, PPO, RBRM  | SFT + RLHF     | MoE               | ✗    | ✓   |
| Llama 3         | [GitHub](https://github.com/meta-llama/llama3)                                                                                                                  | 8B, 70B, 405B       | DPO              | SFT + DPO      | Single Model      | ✓    | ✗   |
| Qwen2           | [GitHub](https://github.com/QwenLM/Qwen2.5)                                                                                                                     | (0.5-72)B, 57B-A14B | DPO              | SFT + DPO      | Single Model      | ✓    | ✓   |
| Gemma2          | [GitHub](https://github.com/google/gemma_pytorch)                                                                                                               | 2B, 9B, 27B         | RLHF             | SFT + RLHF     | Single Model      | ✓    | ✗   |
| Starling-7B     | [GitHub](https://github.com/efrick2002/Starling)                                                                                                                | 7B                  | RLAIF, PPO       | SFT + RLAIF    | Single Model      | ✓    | ✗   |
| Moshi           | [GitHub](https://github.com/kyutai-labs/moshi)                                                                                                                  | 7B                  | -                | -              | Multi-modal       | ✓    | ✓   |
| Athene-70B      | [Hugging Face](https://huggingface.co/Nexusflow/Athene-70B)                                                                                                     | 70B                 | RLHF             | SFT + RLHF     | Single Model      | ✓    | ✗   |
| GPT-3.5         | [GitHub Topics](https://github.com/topics/gpt-4)                                                                                                                | 3.5B, 175B          | RLHF, PPO        | SFT + RLHF     | MoE               | ✗    | ✓   |
| Hermes 3        | [GitHub](https://github.com/meta-llama/llama3)                                                                                                                  | 8B, 70B, 405B       | DPO              | SFT + DPO      | Single Model      | ✓    | ✗   |
| Zed             | [GitHub](https://github.com/zed-industries/zed)                                                                                                                 | 500B                | RLHF             | RLHF           | Multi-modal       | ✓    | ✓   |
| PaLM 2          | [Google AI](https://ai.google/discover/palm2/)                                                                                                                  | -                   | RLHF             | -              | Single Model      | ✗    | ✓   |
| InternLM2       | [GitHub](https://github.com/InternLM/InternLM)                                                                                                                  | 1.8B, 7B, 20B       | RLHF, PPO        | SFT + RLHF     | Single Model      | ✗    | ✗   |
| Supernova       | [GitHub](https://github.com/Nova-Foundation/Supernova)                                                                                                          | 220B                | RLHF             | RLHF           | Multi-modal       | ✓    | ✓   |
| Grok3           | [GitHub](https://github.com/xai-org/grok3)                                                                                                                      | 175B                | -                | DPO            | Dense             | ✓    | ✓   |
| Pixtral         | [GitHub](https://github.com/mistralai/pixtral)                                                                                                                  | 12B, 123B           | -                | PEFT           | Multimodal        | ✓    | ✓   |
| Minimaxtext     | [GitHub](https://github.com/minimax-text/minimaxtext)                                                                                                           | 456B                | -                | SFT            | Single Model      | ✗    | ✗   |
| Amazonnova      | [GitHub](https://github.com/amazon/amazonnova)                                                                                                                  | -                   | DPO, RLHF, RLAIF | SFT            | Single Model      | ✗    | ✗   |
| Fugakullm       | [GitHub](https://github.com/fujitsu/fugakullm)                                                                                                                  | 13B                 | -                | -              | Single Model      | ✗    | ✗   |
| Nova            | [GitHub](https://github.com/rubiks-ai/nova)                                                                                                                     | -                   | -                | SFT            | Proprietary       | ✗    | ✗   |
| 03              | [GitHub](https://github.com/openai/03)                                                                                                                          | -                   | RL through CoT   | RL through CoT | Single Model      | ✗    | ✓   |
| Dbrx            | [GitHub](https://github.com/databricks/dbrx)                                                                                                                    | 136B                | -                | SFT            | Single Model      | ✓    | ✗   |
| Instruct-GPT    | [GitHub](https://github.com/openai/instruct-gpt)                                                                                                                | 1.3B, 6B, 175B      | RLHF, PPO        | SFT + RLHF     | Single Model      | ✗    | ✗   |
| Openassistant   | [GitHub](https://github.com/LAION-AI/Open-Assistant)                                                                                                            | 17B                 | -                | SFT            | Single Model      | ✓    | ✗   |
| ChatGLM         | [GitHub](https://github.com/THUDM/ChatGLM2-6B)                                                                                                                  | 6B, 9B              | ChatGLM-RLHF     | SFT + RLHF     | Single Model      | ✓    | ✗   |
| Zephyr          | [GitHub](https://github.com/huggingface/transformers)                                                                                                           | 141B-A39B           | ORPO             | DPO + ORPO     | MoE               | ✓    | ✓   |
| phi-3           | [GitHub](https://github.com/microsoft/phi-3)                                                                                                                    | 3.8B, 7B, 14B       | DPO              | SFT + DPO      | Single Model      | ✗    | ✗   |
| Jurassic        | [GitHub](https://github.com/ai21labs/jurassic)                                                                                                                  | -                   | -                | SFT            | Proprietary       | ✗    | ✗   |
| Kimi K1.5       | [GitHub](https://github.com/moonshot-ai/kimi)                                                                                                                   | 150B                | -                | RLHF           | Multi-modal       | ✓    | ✓   |
| Phi-4           | [GitHub](https://github.com/microsoft/phi-4)                                                                                                                    | 28B, 70B, 140B      | DPO              | SFT + DPO      | Single Model      | ✗    | ✗   |
| Chameleon       | [GitHub](https://github.com/facebookresearch/chameleon)                                                                                                         | 34B                 | -                | SFT            | Single Model      | ✓    | ✗   |
| Cerebrasgpt     | [GitHub](https://github.com/Cerebras/cerebras-gpt)                                                                                                              | 13B                 | -                | SFT            | Single Model      | ✓    | ✗   |
| Bloomberggpt    | [GitHub](https://github.com/bloomberg/bloomberggpt)                                                                                                             | 50B                 | -                | SFT            | Single Model      | ✗    | ✗   |
| Chinchilla      | [GitHub](https://github.com/deepmind/chinchilla)                                                                                                                | 70B                 | RLHF, PPO        | SFT            | Single Model      | ✗    | ✗   |


# Awesome list 

https://github.com/atfortes/Awesome-LLM-Reasoning


## Survey 

https://arxiv.org/pdf/2501.09686


https://arxiv.org/pdf/2502.21321
## Code 

### Framework
https://github.com/openreasoner/openr/tree/main

https://github.com/THUDM/ReST-MCTS

https://github.com/mys007/ecc

https://github.com/trotsky1997/mathblackbox

https://github.com/natolambert/rlhf-book