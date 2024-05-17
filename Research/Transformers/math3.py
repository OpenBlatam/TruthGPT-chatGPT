import numpy as np

# Assuming the following are predefined
Aj = np.eye(2)  # Sample trainable parameters, for example, an identity matrix
m = 2
Dz = 2  # Length of the task encoding vector

# Feature Wise Linear Modulation functions
γj = np.vectorize(lambda z: z)
βj = np.vectorize(lambda z: 1)

# Example task encoding vector
zi = np.array([1, 1])

# According to the equations given
A_p = np.zeros((m, Dz, Dz))
for j in range(m):
    A_p[j] = Aj * γj(zi) + βj(zi)

# The direct sum operation, equivalent to forming a block-diagonal matrix
M = np.block([np.diag(A_p[j].flatten()) for j in range(m)])

print(M)
