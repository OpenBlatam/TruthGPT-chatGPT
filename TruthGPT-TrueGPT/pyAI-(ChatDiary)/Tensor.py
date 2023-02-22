##Inlcude torch functions , CUDA and printing

class RealTensor(C.RealTensorBase):
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
        assert hasattr(method, typename)
        return getattr(method, typename)(self.size()).copy(self)

    def double(self):
        return self.type('method.DoubleTensor')

    def float(self):
        return self.type('method.FloatTensor')

    def long(self):
        return self.type('method.LongTensor')

    def int(self):
        return self.type('method.IntTensor')

    def short(self):
        return self.type('method.ShortTensor')

    def char(self):
        return self.type('method.CharTensor')

    def byte(self):
        return self.type('method.ByteTensor')
