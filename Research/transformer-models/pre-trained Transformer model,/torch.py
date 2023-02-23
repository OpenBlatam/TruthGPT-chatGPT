import torch
from transformers import BertTokenizer, BertModel

# Load the pre-trained tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the pre-trained model
model = BertModel.from_pretrained('bert-base-uncased')

# Set the model to training mode
model.train()

# Define the input text
input_text = "Hello, how are you doing today?"

# Tokenize the input text
input_ids = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0)

# Pass the input through the model
outputs = model(input_ids)

# Print the outputs
print(outputs)
