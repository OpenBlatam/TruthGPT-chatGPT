import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# Implement openai

# Step 1: Load the dataset
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.tokenizer = tokenizer
        self.examples = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    self.examples.append(self.tokenizer.encode(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx])

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
dataset = TextDataset("data.txt", tokenizer)

# Step 2: Define the model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Step 3: Train the model
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

for epoch in range(5):
    for batch in train_loader:
        inputs, labels = batch[:, :-1], batch[:, 1:]
        loss, logits, _ = model(inputs, labels=labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Step 4: Generate text
prompt = "The quick brown fox"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
