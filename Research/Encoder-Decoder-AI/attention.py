class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        # Dimension of the model
        self.d_model = d_model
        # Depth of each attention head
        self.depth = d_model
        # Linear layer for creating query, key and value matrix
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        # Final linear layer to produce the output
        self.dense = nn.Linear(d_model, d_model)