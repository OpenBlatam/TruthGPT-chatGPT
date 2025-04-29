import triton
import triton.language as tl

@triton.jit
def flash_attention_kernel_4bit(Q_q, K_q, V_q, 
                                Q_scale, Q_min, 
                                K_scale, K_min, 
                                V_scale, V_min,
                                Output, 
                                head_dim, 
                                BLOCK: tl.constexpr):
    start_m = tl.program_id(0)
    start_n = tl.program_id(1)

    q = tl.load(Q_q + start_m * head_dim + tl.arange(0, BLOCK)).to(tl.float32)
    k = tl.load(K_q + start_n * head_dim + tl.arange(0, BLOCK)).to(tl.float32)

    q = q * Q_scale + Q_min
    k = k * K_scale + K_min

    attn_score = tl.sum(q * k, axis=0)
    attn_score = attn_score / tl.sqrt(tl.float32(head_dim))

    # softmax
    attn_exp = tl.exp(attn_score - tl.max(attn_score))
    attn_probs = attn_exp / tl.sum(attn_exp)

    v = tl.load(V_q + start_n * head_dim + tl.arange(0, BLOCK)).to(tl.float32)
    v = v * V_scale + V_min

    output = attn_probs * v

    output_ptr = Output + start_m * head_dim + tl.arange(0, BLOCK)
    tl.store(output_ptr, output)
