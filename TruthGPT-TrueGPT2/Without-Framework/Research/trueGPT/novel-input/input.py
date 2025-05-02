# Import necessary libraries
import random

# Set the random seed
random.seed(42)

# Define a list of prompts
prompts = [
    "The quick brown fox jumps over the lazy dog.",
    "What is the meaning of life?",
    "Tell me a joke.",
    "What is the capital of France?",
    "What is the airspeed velocity of an unladen swallow?",
    "Who is the president of the United States?",
    "How many licks does it take to get to the center of a Tootsie Pop?",
    "What is the answer to the ultimate question of life, the universe, and everything?",
    "What is the largest continent in the world?",
    "What is the square root of 144?"
]

# Choose a random prompt from the list
prompt = random.choice(prompts)

# Print the chosen prompt
print("Prompt: " + prompt)
