import numpy as np
import matplotlib.pyplot as plt

# --- Functions to interpolate ---
f1 = lambda x: np.sin(np.pi * x)          # case 1
f2 = lambda x: 1/(1 + x**2)               # case 2 & 3

# --- Lagrange basis ---
def lagrange_basis(x_points, i, x):
    basis = 1.0
    for j in range(len(x_points)):
        if j != i:
            basis *= (x - x_points[j]) / (x_points[i] - x_points[j])
    return basis

def lagrange_interpolant(x_points, y_points, x):
    p = 0.0
    for i in range(len(x_points)):
        p += y_points[i] * lagrange_basis(x_points, i, x)
    return p

def max_interp_error(f, a, b, n, M=500):
    x_points = np.linspace(a, b, n+1)
    y_points = f(x_points)
    y_samples = np.linspace(a, b, M+1)
    errors = []
    for y in y_samples:
        p_val = lagrange_interpolant(x_points, y_points, y)
        errors.append(abs(f(y) - p_val))
    return max(errors)

# --- Run all cases ---
functions = [(f1, -1, 1, "sin(pi x) on [-1,1]"),
             (f2, -2, 2, "1/(1+x^2) on [-2,2]"),
             (f2, -5, 5, "1/(1+x^2) on [-5,5]")]

n_values = [2,4,8,16]

for f,a,b,label in functions:
    print(f"\nFunction: {label}")
    x_plot = np.linspace(a, b, 500)
    plt.figure(figsize=(8,6))
    plt.plot(x_plot, f(x_plot), 'k-', label="f(x)")
    for n in n_values:
        # error
        err = max_interp_error(f,a,b,n,M=500)
        print(f"  n={n}, max error ≈ {err:.6e}")
        # plot interpolant
        y_points = f(np.linspace(a,b,n+1))
        p_vals = [lagrange_interpolant(np.linspace(a,b,n+1), y_points, xx) for xx in x_plot]
        plt.plot(x_plot, p_vals, label=f"p_{n}(x)")
    plt.title(f"Interpolation of {label}")
    plt.legend()
    plt.show()
