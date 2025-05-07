# The modular design ?

class GPTLayer:
    def __init__(self, input_size, output_size):
        # Initialize layer with weights
        self.weights = torch.randn(input_size, output_size)
        self.previous_functions = []

    def forward(self, x):
        # Example forward pass (e.g., linear transformation)
        self.output = torch.matmul(x, self.weights)
        return self.output

    def backward(self, grad_output):
        # Compute gradients with respect to inputs
        grad_input = torch.matmul(grad_output, self.weights.T)
        grad_weights = torch.matmul(self.input.T, grad_output)
        return grad_input, grad_weights

class GPTModel:
    def __init__(self, input_size, hidden_size, num_layers):
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(GPTLayer(input_size, hidden_size))
            input_size = hidden_size
        self.execution_engine = ExecutionEngine()

    def forward(self, x):
        self.inputs = [x]
        for layer in self.layers:
            x = layer.forward(x)
            self.inputs.append(x)
        return x

    def backward(self, grad_output):
        grad_input = grad_output
        for layer in reversed(self.layers):
            grad_input, grad_weights = layer.backward(grad_input)
            # Use ExecutionEngine to track dependencies
            self.execution_engine.run_backward(layer, grad_weights)
        return grad_input
