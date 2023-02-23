import numpy as np
import multiprocessing

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
    # Create a bigger array with the desired shape and data type
    shape = (10000, 10000)
    dtype = np.float32

    # Use profiling tools to measure performance
    # and identify bottlenecks
    with np.errstate(divide='ignore'):
        # Use parallel processing to create the array
        with multiprocessing.Pool() as pool:
            chunk_shape = (shape[0], shape[1] // multiprocessing.cpu_count())
            chunks = [(chunk_shape, dtype)] * multiprocessing.cpu_count()
            results = pool.starmap(create_array, chunks)
            arr = np.concatenate(results, axis=1)

    # Print the shape and data type of the array
    print("Shape:", arr.shape)
    print("Data type:", arr.dtype)
