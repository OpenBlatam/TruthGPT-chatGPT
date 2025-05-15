# GRPO: Mathematical Formulation

## Objective

The objective function for GRPO is given by:

\[
J_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\} \sim \pi_{\theta_{old}}} \left[ \frac{1}{G} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \min\left(r_t(\theta)A_i, \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_i\right) - \beta D_{KL}[\pi_\theta \|\| \pi_{ref}] \right]
\]

where

\[
r_t(\theta) = \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})}
\]

- $q \sim P(Q)$: question sampled from distribution $P(Q)$
- $\{o_i\} \sim \pi_{\theta_{old}}$: outputs sampled from the old policy
- $G$: number of samples
- $A_i$: advantage estimate for sample $i$
- $\epsilon$: clipping parameter
- $\beta$: KL penalty coefficient
- $D_{KL}[\pi_\theta \|\| \pi_{ref}]$: KL divergence between current and reference policy

---

*This file documents the mathematical formulation for the GRPO algorithm. For implementation, see the corresponding Python files.*



