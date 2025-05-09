import numpy as np

def gauss_elimination(A, b):
    """
    Solves a system of linear equations using Gaussian elimination.

    Args:
      A: A square matrix of coefficients.
      b: A vector of constants.

    Returns:
      A vector of solutions.
    """

    n = len(A)

    # Forward elimination
    for i in range(n):
        # Find the pivot row
        pivot = i
        for j in range(i + 1, n):
            if abs(A[j][i]) > abs(A[pivot][i]):
                pivot = j

        # Swap the pivot row with the current row
        A[i], A[pivot] = A[pivot], A[i]
        b[i], b[pivot] = b[pivot], b[i]

        # Subtract a multiple of the pivot row from the other rows
        for j in range(i + 1, n):
            ratio = -A[j][i] / A[i][i]
            for k in range(i, n + 1):
                A[j][k] += ratio * A[i][k]
            b[j] += ratio * b[i]

    # Back substitution
    x = np.zeros(n)
    x[n - 1] = b[n - 1] / A[n - 1][n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (b[i] - np.dot(A[i][i + 1:], x[i + 1:])) / A[i][i]

    return x
