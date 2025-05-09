import numpy as np

def l2_regularization(loss, weights, regularization_constant):
    """
    Applies L2 regularization to the loss function.

    Args:
        loss: The loss function.
        weights: The weights of the model.
        regularization_constant: The regularization constant.

    Returns:
        The regularized loss function.
    """

    # Calculate the sum of the squared weights.
    total_squared_weights = np.sum(weights**2)

    # Calculate the regularized loss function.
    regularized_loss = loss + regularization_constant * total_squared_weights

    return regularized_loss
