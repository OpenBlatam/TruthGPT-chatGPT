import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PointWiseFeedForwardNetwork(d_model, dff)
        # Layer normalization and dropout layers can be added here

    def forward(self, x):
        attn_output = self.mha(x, x, x)
        out1 = x + attn_output  # Add & Norm
        ffn_output = self.ffn(out1)
        out2 = out1 + ffn_output  # Add & Norm
        return out2
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = PointWiseFeedForwardNetwork(d_model, dff)
        # Layer normalization and dropout layers can be added here

    def forward(self, x, enc_output):
        attn1 = self.mha1(x, x, x)
        out1 = x + attn1  # Add & Norm
        attn2 = self.mha2(out1, enc_output, enc_output)
        out2 = out1 + attn2  # Add & Norm
        ffn_output = self.ffn(out2)
        out3 = out2 + ffn_output  # Add & Norm
        return out3