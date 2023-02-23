import os
import random

def create_gpt_dataset(text_corpus, chunk_size=1024, data_directory="data"):
    """
    Create a dataset for a GPT model from a text corpus.

    Parameters:
    - text_corpus (str): the input text corpus
    - chunk_size (int): the size of each chunk of text (default=1024)
    - data_directory (str): the name of the directory to store the data (default="data")

    Returns:
    - None
    """
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    chunks = [text_corpus[i:i+chunk_size] for i in range(0, len(text_corpus), chunk_size)]
    random.shuffle(chunks)

    for i, chunk in enumerate(chunks):
        filename = os.path.join(data_directory, f"chunk_{i}.txt")
        with open(filename, "w") as f:
            f.write(chunk)

    print(f"Created {len(chunks)} files in directory {data_directory}.")

# Example usage
text_corpus = "This is an example text corpus for training a GPT model. You can use any text data you have available for this purpose."
create_gpt_dataset(text_corpus)
