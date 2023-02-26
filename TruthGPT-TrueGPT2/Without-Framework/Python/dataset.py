import numpy as np
import random
from sklearn.linear_model import LinearRegression

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

# Train a linear regression model on the synthetic data
model = LinearRegression()
model.fit(X, y)

# Make a prediction for a new input
x_new = np.array([[0.5, 0.5]]) # new input feature
y_pred = model.predict(x_new)

print("Input feature:", x_new)
print("Predicted target value:", y_pred)
