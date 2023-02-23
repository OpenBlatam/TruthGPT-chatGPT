import os
import random

def create_gpt_dataset(text_corpus, chunk_size=1024, data_directory="data"):
    # Create the data directory if it doesn't already exist
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    # Split the text corpus into smaller chunks of text
    chunks = [text_corpus[i:i+chunk_size] for i in range(0, len(text_corpus), chunk_size)]

    # Shuffle the chunks to ensure randomness in the training data
    random.shuffle(chunks)

    # Save the chunks to individual text files
    for i, chunk in enumerate(chunks):
        filename = os.path.join(data_directory, f"chunk_{i}.txt")
        with open(filename, "w") as f:
            f.write(chunk)

    print(f"Created {len(chunks)} files in directory {data_directory}")

# Example usage
text_corpus = "This is an example text corpus for training a GPT model. You can use any text data you have available for this purpose."
create_gpt_dataset(text_corpus)
