def backpropagation(X, y, num_hidden, num_iterations, learning_rate, batch_size, reg_lambda):
    # Initialize weights and biases randomly
    num_input = X.shape[1]
    num_output = y.shape[1]
    W1 = np.random.randn(num_input, num_hidden) * np.sqrt(2 / num_input)
    b1 = np.zeros(num_hidden)
    W2 = np.random.randn(num_hidden, num_output) * np.sqrt(2 / num_hidden)
    b2 = np.zeros(num_output)

    # Perform mini-batch gradient descent for each iteration
    num_samples = X.shape[0]
    for i in range(num_iterations):
        # Randomly shuffle the training data
        idx = np.random.permutation(num_samples)
        X_shuffled = X[idx]
        y_shuffled = y[idx]

        # Divide the training data into mini-batches
        for j in range(0, num_samples, batch_size):
            X_batch = X_shuffled[j:j+batch_size]
            y_batch = y_shuffled[j:j+batch_size]

            # Perform forward propagation
            Z1 = X_batch.dot(W1) + b1
            A1 = relu(Z1)
            Z2 = A1.dot(W2) + b2
            y_hat = sigmoid(Z2)

            # Compute the loss and regularization term
            loss = np.mean(-y_batch * np.log(y_hat) - (1 - y_batch) * np.log(1 - y_hat))
            reg_loss = 0.5 * reg_lambda * (np.sum(W1**2) + np.sum(W2**2))
            total_loss = loss + reg_loss

            # Perform backpropagation
            dZ2 = y_hat - y_batch
            dW2 = A1.T.dot(dZ2) + reg_lambda * W2
            db2 = np.sum(dZ2, axis=0)
            dZ1 = dZ2.dot(W2.T) * relu_derivative(Z1)
            dW1 = X_batch.T.dot(dZ1) + reg_lambda * W1
            db1 = np.sum(dZ1, axis=0)

            # Update weights and biases
            W2 -= learning_rate * dW2 / batch_size
            b2 -= learning_rate * db2 / batch_size
            W1 -= learning_rate * dW1 / batch_size
            b1 -= learning_rate * db1 / batch_size

    return W1, b1, W2, b2
