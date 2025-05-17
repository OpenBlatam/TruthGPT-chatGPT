MCTS-AHD
MCTS-AHD (also referred to as MCTSAHD) normalizes the quality value Q(·) to enhance the homogeneity of different tasks when calculating the UCT value for each child node c ∈ Children(n_c) of a node n_c, as follows:


UCT(c) = (Q(c) - q_min) / (q_max - q_min) + λ * sqrt(ln(N(n_c) + 1) / N(c))

Where:

q_max and q_min are the maximum and minimum quality values Q(·) encountered during the MCTS process,

λ is a tunable exploration parameter,

N(n_c) is the visit count of node n_c,

N(c) is the visit count of child node c.

From the root node n_r, MCTS iteratively selects the child node with the highest UCT value until reaching a leaf node.

advantage: a more
granular, token-level reward modeling framework.


# Code

https://github.com/zz1358m/MCTS-AHD-master/tree/main

https://github.com/sabijun/MT-RewardTree/tree/main