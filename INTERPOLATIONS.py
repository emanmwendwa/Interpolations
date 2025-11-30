import numpy as np

# 1. Lagrange basis function ℓ_i(x)
def lagrange_basis(x_points, i, x):
    """Compute Lagrange basis ℓ_i(x) for given x."""
    basis = 1.0
    for k in range(len(x_points)):
        if k != i:
            basis *= (x - x_points[k]) / (x_points[i] - x_points[k])
    return basis

#2: interpolating polynomial p_n(x)
def lagrange_interpolant(x_points, y_points, x):
    """Evaluate interpolating polynomial at x."""
    n = len(x_points)
    p = 0.0
    for i in range(n):
        p += y_points[i] * lagrange_basis(x_points, i, x)
    return p

#3: Approximate max interpolation error
def max_interp_error(f, a, b, n, m=500):
    """Compute max error between f(x) and p_n(x) on [a,b]."""
    # Interpolation points
    x_points = np.linspace(a, b, n+1)
    y_points = f(x_points)

    # Sampling points
    y_samples = np.linspace(a, b, m + 1)
    errors = []
    for y in y_samples:
        p_val = lagrange_interpolant(x_points, y_points, y)
        errors.append(abs(f(y) - p_val))

    return max(errors)

# Define test function
f1 = lambda x: np.sin(np.pi * x)

# Compute error for n=2
error = max_interp_error(f1, -1, 1, n=2, m=500)
print("Approximate max error (sin(pi x), n=2):", error)

