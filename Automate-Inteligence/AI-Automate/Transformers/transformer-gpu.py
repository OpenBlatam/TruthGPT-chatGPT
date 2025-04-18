import tensorflow as tf

def attention(Q, K, V):
    dk = tf.shape(Q)[-1]
    QKT = tf.linalg.matmul(Q, K, transpose_b=True)
    weighted_values = tf.linalg.matmul(tf.nn.softmax(QKT / tf.math.sqrt(dk.astype(float)), axis=-1), V)
    return weighted_values

# Define the Q, K, V here
Q = tf.random.uniform(shape=(2,2), dtype=tf.float32)
K = tf.random.uniform(shape=(2,2), dtype=tf.float32)
V = tf.random.uniform(shape=(2,2), dtype=tf.float32)

with tf.device('/GPU:0'):  # Use GPU if available, otherwise use CPU
    # Call the function
    outputs = attention(Q, K, V)