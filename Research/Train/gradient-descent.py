import numpy as np

def gradient_descent(func, grad, x0, alpha=0.1, num_iterations=100):
    x = x0.copy()
    for i in range(num_iterations):
        gradient = grad(x)
        x -= alpha * gradient
        cost = func(x)
        print(f"Iteration {i}, cost = {cost}")
    return x

# Define the function to be optimized
def func(x):
    return x[0]**2 + x[1]**2

# Define the gradient of the function
def grad(x):
    return np.array([2*x[0], 2*x[1]])

# Initialize the parameters
x0 = np.array([1, 1])

# Run the gradient descent algorithm
x_opt = gradient_descent(func, grad, x0, alpha=0.1, num_iterations=100)

# Print the final parameters
print(f"Final parameters: {x_opt}")
