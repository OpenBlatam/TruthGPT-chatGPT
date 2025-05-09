import sympy as sp

# Define symbols
x = sp.Symbol('x')  # Input
w1, b1 = sp.symbols('w1 b1')  # First layer weights and bias
w2, b2 = sp.symbols('w2 b2')  # Second layer weights and bias

# Define the activation function (sigmoid approx): sigmoid(z) ≈ 0.5 + z/4 - z**3/48
z = w1 * x + b1
sigma_approx = 0.5 + z/4 - z**3/48

# Output of MLP: y = w2 * sigma(z) + b2
y = w2 * sigma_approx + b2

# Simplify final polynomial
y_poly = sp.simplify(sp.expand(y))
y_poly
