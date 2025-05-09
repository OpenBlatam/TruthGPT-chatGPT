import numpy as np

class RealTensor:
    def __init__(self, data):
        self.data = np.array(data)

    def __str__(self):
        return "RealTensor"

    def __repr__(self):
        return str(self)

    def type(self, t):
        current = "method." + self.__class__.__name__
        if not t:
            return current
        if t == current:
            return self
        _, _, typename = t.partition('.')
        assert hasattr(self, typename)
        return getattr(self, typename)()

    def double(self):
        return RealTensor(self.data.astype(np.float64))

    def float(self):
        return RealTensor(self.data.astype(np.float32))

    def long(self):
        return RealTensor(self.data.astype(np.int64))

    def int(self):
        return RealTensor(self.data.astype(np.int32))

    def short(self):
        return RealTensor(self.data.astype(np.int16))

    def char(self):
        return RealTensor(self.data.astype(np.int8))

    def byte(self):
        return RealTensor(self.data.astype(np.uint8))
