import numpy as np

# Triangular solvers (Forward and Back Substitution)

def forward_sub(L, b):
    # Solve lower triangular system Lx = b
    n = len(b)
    x = np.zeros(n, dtype=np.float64)
    for i in range(n):
        # x[i] = (b[i] - sum(L[i,j]*x[j])) / L[i,i]
        sum_part = 0
        for j in range(i):
            sum_part += L[i, j] * x[j]
        x[i] = (b[i] - sum_part) / L[i, i]
    return x

def back_sub(U, b):
    # Solve upper triangular system Ux = b
    n = len(b)
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        # x[i] = (b[i] - sum(U[i,j]*x[j])) / U[i,i]
        sum_part = 0
        for j in range(i + 1, n):
            sum_part += U[i, j] * x[j]
        x[i] = (b[i] - sum_part) / U[i, i]
    return x
