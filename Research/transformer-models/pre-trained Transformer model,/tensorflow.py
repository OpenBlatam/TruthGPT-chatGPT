import tensorflow as tf
from transformers import TFDistilBertModel, DistilBertTokenizer

# Load the pre-trained tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load the pre-trained model
model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

# Define the input text
input_text = "Hello, how are you doing today?"

# Tokenize the input text
input_ids = tokenizer.encode(input_text, add_special_tokens=True, return_tensors='tf')

# Pass the input through the model
outputs = model(input_ids)

# Print the outputs
print(outputs)
