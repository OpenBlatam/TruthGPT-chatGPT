import tensorflow as tf

# Assuming you have input and output sequences as NumPy arrays
input_seqs = ...  # Your input sequences as a NumPy array
output_seqs = ...  # Your output sequences as a NumPy array

# Convert the NumPy arrays to TensorFlow tensors
input_seqs = tf.convert_to_tensor(input_seqs)
output_seqs = tf.convert_to_tensor(output_seqs)

# Create the training data using tf.data.Dataset.from_tensor_slices
training_data = tf.data.Dataset.from_tensor_slices((input_seqs, output_seqs))

# Further preprocessing or batching steps can be applied to the training data
training_data = training_data.batch(batch_size)
