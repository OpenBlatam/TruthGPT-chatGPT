# Create a training dataset

training_dataset = [['This', 'is', 'a', 'sentence'],
                    ['I', 'am', 'a', 'model'],
                    ['The', 'model', 'is', 'good']]

# Create a model
model = autoregressive_tranformer.AutoregressiveTransformer(input_size=4, hidden_size=16, output_size=4)

# Train the model
model.train(training_dataset, epochs=10)

# Generate text
generated_text = model.generate('This is ')

# Print the generated text
print(generated_text)
