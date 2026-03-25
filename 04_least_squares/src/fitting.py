import numpy as np

# A-Matrix construction for different Least Squares models

def linear_model(x_data):
    # Model: y = c0 + c1*x
    A = np.ones((len(x_data), 2))
    A[:, 1] = x_data
    return A

def polynomial_model(x_data, order):
    # Model: y = c0 + c1*x + c2*x^2 + ...
    A = np.ones((len(x_data), order + 1))
    for i in range(1, order + 1):
        A[:, i] = x_data ** i
    return A

def exponential_model(x_data):
    # Model: ln(y) = ln(a) + b * x
    # This just returns the linear design matrix because we fit in log-space
    return linear_model(x_data)
