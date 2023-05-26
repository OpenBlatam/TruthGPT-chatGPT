from sklearn.datasets import load_iris, load_digits
from sklearn.datasets import fetch_openml

# Load Iris dataset
iris = load_iris()
iris_data = iris.data
iris_target = iris.target

# Load MNIST digits dataset
digits = load_digits()
digits_data = digits.data
digits_target = digits.target

# Load Boston housing dataset
boston = fetch_openml(name='boston', version=1, as_frame=True)
boston_data = boston.data
boston_target = boston.target

# Print the shape of the loaded datasets
print("Iris dataset shape:", iris_data.shape)
print("Digits dataset shape:", digits_data.shape)
print("Boston dataset shape:", boston_data.shape)
