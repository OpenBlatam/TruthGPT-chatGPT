import numpy as np

def create_array(shape, dtype):
    """
    Creates a NumPy array with the specified shape and data type,
    with all elements initialized to 0.0.

    Args:
        shape: A tuple representing the shape of the array.
        dtype: The data type of the array.

    Returns:
        A NumPy array with the specified shape and data type,
        with all elements initialized to 0.0.
    """
    # Use a view to set all elements to 0.0
    arr = np.zeros(shape, dtype=dtype)
    view = arr.reshape(-1)
    view[:] = 0.0

    return arr

if __name__ == '__main__':
    # Create an array with the desired shape and data type
    shape = (1000, 1000)
    dtype = np.float64

    # Use profiling tools to measure performance
    # and identify bottlenecks
    with np.errstate(divide='ignore'):
        arr = create_array(shape, dtype)

    # Print the array
    print(arr)
