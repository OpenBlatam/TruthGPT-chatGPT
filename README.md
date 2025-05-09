TruthGPT
============

There are several guides for ML and AI developers and users. These guides can
be rendered in a number of formats, like Models and Transformers. Please read
Documentation/admin-guide/README.rst first.

There are various text files in the Documentation/ subdirectory,
several of them using the Restructured Text markup notation.

Please read the Documentation/process/changes.rst file, as it contains the
requirements for building and running the GPT, and information about
the problems which may result by upgrading your GPT.

## Mathematical Notation

Let $X$ be the space of possible queries (e.g., user prompts). For each query $x \in X$, we collect one or more candidate responses $\{y_j\}_{j=1}^{m_x}$ where $m_x$ is the number of candidate responses for query $x$.

The dataset $D$ is defined as:
$$D = \{(x_i, \{y_{ij}\}_{j=1}^{m_i}, \{\text{preferences}_i\})\}_{i=1}^N$$

## Reinforcement Learning Framework

Once we have a trained reward model $R_\theta(x, y)$ that captures human preferences, we can integrate it into a RL framework to optimize a policy $\pi_\phi$. In essence, we replace (or augment) the environment's native reward signal with $R_\theta(x, y)$ so that the agent focuses on producing responses $y$ that humans prefer for a given query $x$.

In typical RL notation:
- Each state $s$ here can be interpreted as the partial dialogue or partial generation process for the next token (in language modeling).
- Each action $a$ is the next token (or next chunk of text) to be generated.
- The policy $\pi_\phi(a | s)$ is a conditional distribution over the next token, parameterized by $\phi$.

We seek to find $\phi$ that maximizes the expected reward under $R_\theta$. Concretely, let $x$ be a user query, and let $y \sim \pi_\phi(\cdot | x)$ be the generated response. We aim to solve:

$$\max_\phi \mathbb{E}_{x \sim X} \left[ \mathbb{E}_{y \sim \pi_\phi(\cdot | x)} \left[ R_\theta(x, y) \right] \right]$$

This means that on average, over user queries $x$ and responses $y$ drawn from the policy $\pi_\phi$, we want the reward model's score $R_\theta(x, y)$ to be as high as possible.
