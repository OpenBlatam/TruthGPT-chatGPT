import re

def my_tokenizer(text):
    # Replace any non-alphanumeric characters with a space
    text = re.sub(r'\W+', ' ', text)

    # Split the text into tokens
    tokens = text.split()

    return tokens
