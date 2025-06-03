import numpy as np


def softmax(x):
    """Compute softmax values for array x."""
    if len(x) == 0:
        return np.array([])
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x)


class FineTuner:
    """A class for fine-tuning operations."""
    
    def __init__(self):
        self.n = None
        self.learning_rate = 0.001
        self.epochs = 10
    
    def finetune(self, n, data=None, learning_rate=None):
        """
        Fine-tune the model with given parameters.
        
        Args:
            n: Number of parameters or dimension
            data: Training data (optional)
            learning_rate: Learning rate for fine-tuning (optional)
        
        Returns:
            dict: Fine-tuning results
        """
        self.n = n
        
        if learning_rate is not None:
            self.learning_rate = learning_rate
        
        # Simulate fine-tuning process
        if data is not None:
            # Apply softmax to data if it's provided
            if isinstance(data, (list, np.ndarray)):
                data_array = np.array(data)
                if data_array.ndim == 1:
                    softmax_result = softmax(data_array)
                else:
                    # Apply softmax to each row
                    softmax_result = np.array([softmax(row) for row in data_array])
            else:
                softmax_result = None
        else:
            softmax_result = None
        
        # Return fine-tuning results
        return {
            'n': self.n,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'softmax_result': softmax_result,
            'status': 'completed'
        }
    
    def set_hyperparameters(self, learning_rate=None, epochs=None):
        """Set hyperparameters for fine-tuning."""
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if epochs is not None:
            self.epochs = epochs
    
    def get_hyperparameters(self):
        """Get current hyperparameters."""
        return {
            'learning_rate': self.learning_rate,
            'epochs': self.epochs
        }


# Standalone function for backward compatibility
def finetune(n, data=None, learning_rate=0.001):
    """
    Standalone fine-tune function.
    
    Args:
        n: Number of parameters or dimension
        data: Training data (optional)
        learning_rate: Learning rate for fine-tuning
    
    Returns:
        dict: Fine-tuning results
    """
    tuner = FineTuner()
    return tuner.finetune(n, data, learning_rate)
    
