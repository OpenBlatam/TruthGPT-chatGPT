📘 LDPO (Logit-based Direct Preference Optimization) Loss Function
The LDPO loss function is defined as:

\[
\mathcal{L}_{\text{LDPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]
\]

Where:

𝑥
x: Input prompt.

𝑦
𝑤
y 
w
​
 : Preferred (winning) response.

𝑦
𝑙
y 
l
​
 : Less preferred (losing) response.

𝜋
𝜃
π 
θ
​
 : Current policy parameterized by 
𝜃
θ.

𝜋
ref
π 
ref
​
 : Reference policy.

𝛽
β: Scaling factor.

𝜎
σ: Sigmoid function.

Explanation:

This loss function encourages the model to assign higher probabilities to preferred responses compared to less preferred ones. By comparing the log-probabilities of the current policy and a reference policy for both preferred and less preferred responses, and applying the sigmoid function, the model is guided to align its outputs with human preferences.
## Survey

https://arxiv.org/pdf/2503.11701



| Model         | Parameters         | RL Methods        | Fine-Tuning         | Architecture | Open-Source | Reasoning Focus           | GitHub Repository                                                                 |
|---------------|--------------------|-------------------|---------------------|--------------|-------------|---------------------------|-----------------------------------------------------------------------------------|
| DeepSeek-R1   | 240B (MoE), 22B    | GRPO              | DPO + GRPO          | MoE          | ✅ Yes      | General reasoning         | [deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)             |
| InternLM-Math | 1.8B, 7B, 20B      | RLHF, PPO         | SFT + RLHF          | Single Model | ✅ Yes      | Mathematical reasoning    | [InternLM/InternLM-Math](https://github.com/InternLM/InternLM-Math)               |
| Zephyr        | 141B (MoE), 39B    | ORPO              | DPO + ORPO          | MoE          | ✅ Yes      | General, diverse tasks    | [sanjaybip/llm-zephyr-langchain-chainlit](https://github.com/sanjaybip/llm-zephyr-langchain-chainlit) |


Notes:

MoE: Mixture of Experts architecture, which allows models to scale efficiently by activating only a subset of parameters during inference.

GRPO: Generalized Reinforcement Policy Optimization.

DPO: Direct Preference Optimization.

RLHF: Reinforcement Learning from Human Feedback.

PPO: Proximal Policy Optimization.

ORPO: Open Reinforcement Preference Optimization.

SFT: Supervised Fine-Tuning.