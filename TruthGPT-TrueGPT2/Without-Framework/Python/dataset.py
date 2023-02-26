import numpy as np
import random

def generate_regression_data(num_samples, num_features):
    # Generate random input features
    X = np.random.rand(num_samples, num_features)

    # Generate target values using a linear function
    true_coef = np.random.rand(num_features)
    true_intercept = np.random.rand()
    noise = np.random.normal(scale=0.1, size=num_samples)
    y = np.dot(X, true_coef) + true_intercept + noise

    return X, y

# Generate a synthetic dataset with 100 samples and 2 features
X, y = generate_regression_data(100, 2)
