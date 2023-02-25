import torch

class RealTensor(torch.Tensor):
    def __str__(self):
        return "RealTensor"

    def __repr__(self):
        return str(self)

    def type(self, t):
        current = "RealTensor"
        if not t:
            return current
        if t == current:
            return self
        _, _, typename = t.partition('.')
        assert hasattr(torch, typename + 'Tensor')
        return getattr(torch, typename + 'Tensor')(self.size()).copy_(self)

    def double(self):
        return self.type('torch.DoubleTensor')

    def float(self):
        return self.type('torch.FloatTensor')

    def long(self):
        return self.type('torch.LongTensor')

    def int(self):
        return self.type('torch.IntTensor')

    def short(self):
        return self.type('torch.ShortTensor')

    def char(self):
        return self.type('torch.CharTensor')

    def byte(self):
        return self.type('torch.ByteTensor')

# Example usage
x = RealTensor([1, 2, 3])
print(x)
y = x.double()
print(y)
z = x.to('cuda')
print(z)
