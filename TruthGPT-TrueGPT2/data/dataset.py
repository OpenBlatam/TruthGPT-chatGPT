import os
import random

data_directory = "data"
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

# Define your text corpus here
text_corpus = "This is an example text corpus for training a GPT model. You can use any text data you have available for this purpose."

# Split the text corpus into smaller chunks of text
chunk_size = 1024
chunks = [text_corpus[i:i+chunk_size] for i in range(0, len(text_corpus), chunk_size)]

# Shuffle the chunks to ensure randomness in the training data
random.shuffle(chunks)

# Save the chunks to individual text files
for i, chunk in enumerate(chunks):
    filename = os.path.join(data_directory, f"chunk_{i}.txt")
    with open(filename, "w") as f:
        f.write(chunk)
