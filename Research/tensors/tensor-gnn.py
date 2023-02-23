import numpy as np

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class PositionalEncoding:
    def __init__(self, d_model, max_len):
        self.pos_enc = np.zeros((max_len, d_model))
        pos = np.arange(0, max_len)[:, np.newaxis]
        div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        self.pos_enc[:, 0::2] = np.sin(pos * div)
        self.pos_enc[:, 1::2] = np.cos(pos * div)

    def __call__(self, x):
        return x + self.pos_enc[:x.shape[1], :]

class MultiHeadAttention:
    def __init__(self, n_heads, d_model):
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_layer = LinearLayer(d_model, d_model)
        self.k_layer = LinearLayer(d_model, d_model)
        self.v_layer = LinearLayer(d_model, d_model)
        self.out_layer = LinearLayer(d_model, d_model)

    def __call__(self, q, k, v, mask=None):
        bs = q.shape[0]
        q = self.q_layer(q).reshape(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_layer(k).reshape(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_layer(v).reshape(bs, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = np.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        attn_weights = softmax(scores)
        attn_output = np.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).reshape(bs, -1, self.n_heads * self.d_k)
        return self.out_layer(attn_output)

class FeedForwardLayer:
    def __init__(self, d_model, d_ff):
        self.fc1 = LinearLayer(d_model, d_ff)
        self.fc2 = LinearLayer(d_ff, d_model)

    def __call__(self, x):
        x = self.fc1(x)
        x = np.maximum(x, 0.0)
        x = self.fc2(x)
        return x
class Encoder:
    def __init__(self, n_layers, n_heads, d_model, d_ff, max_len):
        self.layers = [EncoderLayer(n_heads, d_model, d_ff) for _ in range(n_layers)]
        self.pe = PositionalEncoding(d_model, max_len)

    def __call__(self, x, mask=None):
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x
