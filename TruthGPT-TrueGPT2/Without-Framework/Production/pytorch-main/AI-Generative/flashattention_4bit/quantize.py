import torch

def quantize_4bit(x):
    """Quantize a tensor to 4-bit unsigned integers."""
    x_min = x.min()
    x_max = x.max()

    scale = (x_max - x_min) / 15.0
    x_q = ((x - x_min) / scale).round().clamp(0, 15).to(torch.uint8)

    return x_q, scale, x_min

def dequantize_4bit(x_q, scale, x_min):
    """Dequantize 4-bit tensor to float32."""
    return x_q.to(torch.float32) * scale + x_min
