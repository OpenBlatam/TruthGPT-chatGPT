import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x.float()**2, dim=-1, keepdim=True) + self.eps)
        return x * (self.weight / rms).to(x.dtype)

def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    seq_len = xq.shape[1]
    cos = freqs_cos[:seq_len].unsqueeze(0).unsqueeze(2)
    sin = freqs_sin[:seq_len].unsqueeze(0).unsqueeze(2)
    xq_rot = xq.float()
    xk_rot = xk.float()
    xq1, xq2 = xq_rot.chunk(2, dim=-1)
    xk1, xk2 = xk_rot.chunk(2, dim=-1)
    xq_out = torch.cat([xq1 * cos - xq2 * sin, xq1 * sin + xq2 * cos], dim=-1)
    xk_out = torch.cat([xk1 * cos - xk2 * sin, xk1 * sin + xk2 * cos], dim=-1)
    return xq_out.to(xq.dtype), xk_out.to(xk.dtype)

class Attention(nn.Module):
    def __init__(self, dim, num_heads, max_seq_len):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        freqs_cos, freqs_sin = precompute_freqs_cis(self.head_dim, max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)
    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.num_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.num_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.num_heads, self.head_dim)
        q, k = apply_rotary_emb(q, k, self.freqs_cos, self.freqs_sin)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(attn_output)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.gelu = nn.GELU()
    def forward(self, x):
        return self.w2(self.gelu(self.w1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, max_seq_len):
        super().__init__()
        self.attention = Attention(dim, num_heads, max_seq_len)
        self.ff = FeedForward(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
    def forward(self, x, mask=None):
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x

class S(nn.Module):
    def __init__(self, vocab_size=32000, dim=512, num_layers=12, num_heads=8, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim, num_heads, max_seq_len) for _ in range(num_layers)])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits

def get_model():
    return S(vocab_size=32000, dim=512, num_layers=12, num_heads=8, max_seq_len=2048)
