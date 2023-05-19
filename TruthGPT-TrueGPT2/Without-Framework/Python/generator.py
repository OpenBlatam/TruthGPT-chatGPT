import random

class NPLGenerator:
  def __init__(self, n):
    self.n = n
    self.alphabet = "abcdefghijklmnopqrstuvwxyz"

  def generate(self):
    npl = ""
    for i in range(self.n):
      npl += random.choice(self.alphabet)
    return npl

for i in range(10):
  print(NPLGenerator(10).generate())
