import tensorflow as tf

# Create a sparse tensor directly
sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 1], [2, 0], [2, 2]],
                                       values=[4.0, 5.0, 6.0],
                                       dense_shape=[3, 3])

print("Sparse tensor:")
print(sparse_tensor)
