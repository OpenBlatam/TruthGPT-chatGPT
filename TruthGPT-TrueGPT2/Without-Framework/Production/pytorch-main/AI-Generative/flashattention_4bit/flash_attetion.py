import torch
from .quantize import quantize_4bit
from .triton import flash_attention_kernel_4bit

class FlashAttention4bit(torch.nn.Module):
    def __init__(self, head_dim, block_size=64):
        super().__init__()
        self.head_dim = head_dim
        self.block_size = block_size

    def forward(self, Q, K, V):
        # Quantize
        Q_q, Q_scale, Q_min = quantize_4bit(Q)
        K_q, K_scale, K_min = quantize_4bit(K)
        V_q, V_scale, V_min = quantize_4bit(V)

        Output = torch.empty_like(Q, dtype=torch.float32)

        flash_attention_kernel_4bit[(Q.shape[2], K.shape[2])](
            Q_q.view(-1, self.head_dim),
            K_q.view(-1, self.head_dim),
            V_q.view(-1, self.head_dim),
            Q_scale, Q_min, 
            K_scale, K_min,
            V_scale, V_min,
            Output.view(-1, self.head_dim),
            self.head_dim,
            BLOCK=self.block_size
        )

        return Output
