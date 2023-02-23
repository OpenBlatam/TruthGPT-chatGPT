from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(lower=True)

def tokenize(text):
    tokenizer.fit_on_texts([text])
    tokens = tokenizer.texts_to_sequences([text])[0]
    tokens = [tokenizer.index_word[token] for token in tokens]
    return tokens
