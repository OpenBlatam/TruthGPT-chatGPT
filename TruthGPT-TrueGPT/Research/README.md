## Research
Goal:
Running minimize the spec or implementation for looking in benchmarks.

```python 
import numpy as np

def gauss_elim(A, b):
    n = len(A)
    for i in range(n):
        for j in range(i+1, n):
            factor = A[j][i] / A[i][i]
            A[j] -= factor * A[i]
            b[j] -= factor * b[i]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i][i+1:], x[i+1:])) / A[i][i]
    return x

A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]])
b = np.array([8, -11, -3])
x = gauss_elim(A, b)
print(x)

```
How to run: query the AI

Thread: query AI in multi AI connect if non trivial? 


Research
This repository is used mainly for code related to specific research questions, mostly written by . It is not meant as a general research repository for academic papers.

An exception to this is the papers folder, which contains the LaTeX files for various academic papers.

Contribute
While contributions are welcome, maintaining this repository is not an active priority. The code in this repository is offered as is, without active support.

If you find spelling errors or have suggestions or comments, please feel free to open an issue.
