import torch

class RealTensor(torch.Tensor):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __str__(self):
        return "RealTensor"

    def __repr__(self):
        return str(self)

    def type(self, dtype=None, non_blocking=False, **kwargs):
        if dtype is None:
            return self
        return super().type(dtype, non_blocking=non_blocking, **kwargs)

# Example usage
x = RealTensor([1, 2, 3])
print(x)
y = x.double()
print(y)
z = x.to('cuda')
print(z)
