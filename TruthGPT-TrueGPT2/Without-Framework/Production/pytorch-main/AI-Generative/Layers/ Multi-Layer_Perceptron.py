

import torch
import torch.nn as nn
import torch.nn.functional as F
# Extract Transformer for hugging face?
class MLP(nn.Module):
    """
    ulti-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.M
    """
 