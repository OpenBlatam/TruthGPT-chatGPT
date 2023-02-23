import string

def tokenize(text):
    # Convert to lowercase
    text = text.lower()

    # Replace all punctuation marks with spaces
    for char in string.punctuation:
        text = text.replace(char, " ")

    # Split on whitespace
    tokens = text.split()

    return tokens
