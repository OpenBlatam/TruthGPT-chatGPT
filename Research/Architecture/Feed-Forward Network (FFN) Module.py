from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the model
model = Sequential()

# Add input layer
# 'units' indicates the number of neurons in the first hidden layer
# 'input_dim' signifies the number of inputs
# 'activation' is the activation function for the neurons
model.add(Dense(units=32, input_dim=784, activation='relu'))

# Add Output layer
# 'units' here indicates the number of neurons in the output layer.
# It should match the number of unique classifications for the problem
model.add(Dense(units=10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print a summary of your model
model.summary()