import re

def tokenize(text):
    # Split on non-alphanumeric characters and convert to lowercase
    tokens = re.split(r'\W+', text.lower())

    return tokens
