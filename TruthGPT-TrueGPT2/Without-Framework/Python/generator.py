import random
import transformers

class NPLGenerator:
  def __init__(self, n):
    self.n = n
    self.model = transformers.GPT3LMHeadModel.from_pretrained("gpt3")

  def generate(self):
    return ''.join(random.choices(self.model.vocab, k=self.n))

for i in range(10):
  print(NPLGenerator(10).generate())
