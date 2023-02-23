def generate_sentence(prompt, max_sequence_length=10):
    h = [0.0] * hidden_units
    x = [word_to_index[word] for word in prompt.split()]
    generated_sentence = prompt
    for i in range(max_sequence_length):
        y = forward(x)
        next_word = np.random.choice(range(vocab_size), p=y)
        if next_word == 0: # end-of-sequence token
            break
        generated_sentence += " " + vocab[next_word]
        x = [next_word]
        h = [np.tanh(sum([Wxh[i][j] * x[0] + Whh[i][j] * h[j] for j in range(hidden_units)])) for i in range(hidden_units)]
    return generated_sentence
