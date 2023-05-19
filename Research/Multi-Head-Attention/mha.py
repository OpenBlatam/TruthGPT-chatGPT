import torch

class OptimizedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.q_linear = nn.Linear(d_model, d_model * num_heads // 2)
        self.k_linear = nn.Linear(d_model, d_model * num_heads // 2)
        self.v_linear = nn.Linear(d_model, d_model * num_heads // 2)

        self.attention = nn.MultiheadAttention(num_heads, d_model // 2)

        self.output_linear = nn.Linear(d_model * num_heads // 2, d_model)

    def forward(self, query, key, value):
        """
        query: [batch_size, seq_len, d_model]
        key: [batch_size, seq_len, d_model]
        value: [batch_size, seq_len, d_model]

        Returns:
        output: [batch_size, seq_len, d_model]
        """

        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value)

        query = query.view(query.size(0), query.size(1), self.num_heads, query.size(2) // self.num_heads)
        key = key.view(key.size(0), key.size(1), self.num_heads, key.size(2) // self.num_heads)
        value = value.view(value.size(0), value.size(1), self.num_heads, value.size(2) // self.num_heads)

        attention_output = self.attention(query, key, value)

        attention_output = attention_output.view(attention_output.size(0), attention_output.size(1), self.num_heads * (attention_output.size(2) // self.num_heads))

        output = self.output_linear(attention_output)

        return output
