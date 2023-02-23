import numpy as np

def stochastic_gradient_descent(X, y, alpha=0.01, epochs=100, batch_size=32):
    """
    Performs stochastic gradient descent on the given data.

    Parameters:
    X (np.ndarray): The feature matrix.
    y (np.ndarray): The target vector.
    alpha (float): The learning rate.
    epochs (int): The number of epochs to train for.
    batch_size (int): The size of each mini-batch.

    Returns:
    w (np.ndarray): The learned weights.
    b (float): The learned bias.
    losses (list): A list of loss values for each epoch.
    """
    n, m = X.shape
    w = np.zeros((m, 1))
    b = 0
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0

        # Shuffle the data
        permutation = np.random.permutation(n)
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]

        for i in range(0, n, batch_size):
            # Get the mini-batch
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # Compute the gradient
            y_pred = np.dot(X_batch, w) + b
            error = y_batch.reshape((-1, 1)) - y_pred
            grad_w = -2 * np.dot(X_batch.T, error) / batch_size
            grad_b = -2 * np.sum(error) / batch_size

            # Update the parameters
            w -= alpha * grad_w
            b -= alpha * grad_b

            # Compute the loss for this mini-batch
            mini_batch_loss = np.mean(error**2)
            epoch_loss += mini_batch_loss

        # Compute the loss for this epoch
        epoch_loss /= (n // batch_size)
        losses.append(epoch_loss)

        print(f"Epoch {epoch+1}/{epochs}: loss={epoch_loss}")

    return w, b, losses
