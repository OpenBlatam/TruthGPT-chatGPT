import torchtext

tokenizer = torchtext.data.utils.get_tokenizer("spacy", language="en_core_web_sm")
TEXT = torchtext.data.Field(lower=True, tokenize=tokenizer, batch_first=True)

def tokenize(text):
    preprocessed = TEXT.preprocess(text)
    tokens = preprocessed[0]
    return tokens

text = "Hello, world! This is a sample text for tokenization."
tokens = tokenize(text)
print(tokens)
