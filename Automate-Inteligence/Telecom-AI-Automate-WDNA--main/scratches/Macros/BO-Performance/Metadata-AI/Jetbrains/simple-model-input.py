from transformers import pipeline, set_seed
import requests
import json

# load pre-trained model and create generator with it
generator = pipeline('text-generation', model='gpt2')

set_seed(42)

# You would replace this with actual ticket text
ticket_text = "The user has issues connecting to the internet"

# Generate text with the model
outputs = generator(ticket_text, max_length=100, num_return_sequences=1)

print(outputs)