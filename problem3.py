import numpy as np
import matplotlib.pyplot as plt

def lagrange_basis_eval(x_nodes, i, x):
    """
    Evaluate the Lagrange basis l_i(x) for node i over points x.
    x_nodes: array of interpolation nodes (length n+1)
    i: index of the basis (0..n)
    x: array-like points at which to evaluate
    """
    xi = x_nodes[i]
    # Product over k != i of (x - xk)/(xi - xk)
    denom = np.prod([(xi - x_nodes[k]) for k in range(len(x_nodes)) if k != i])
    # Vectorized numerator
    num = np.ones_like(x, dtype=float)
    for k in range(len(x_nodes)):
        if k != i:
            num *= (x - x_nodes[k])
    return num / denom

def lagrange_interpolant(x_nodes, f_nodes, x_eval):
    """
    Evaluate Lagrange interpolant at x_eval given nodes and values.
    """
    n = len(x_nodes) - 1
    p = np.zeros_like(x_eval, dtype=float)
    for i in range(n+1):
        li = lagrange_basis_eval(x_nodes, i, x_eval)
        p += f_nodes[i] * li
    return p

def run_case(f, a, b, ns=(2,4,8,16), M=500, title=None, do_plot=True):
    """
    For function f on [a,b], compute max errors using n in ns and sampling M+1 points.
    Returns dict {n: approx_max_error}.
    Optionally plots f and p_n curves.
    """
    y = np.linspace(a, b, M+1)
    fy = f(y)
    results = {}
    if do_plot:
        plt.figure(figsize=(10,6))
        plt.plot(y, fy, 'k-', label='f(x)', linewidth=2)

    for n in ns:
        x_nodes = np.linspace(a, b, n+1)
        f_nodes = f(x_nodes)
        py = lagrange_interpolant(x_nodes, f_nodes, y)
        err = np.abs(fy - py)
        approx_max_err = np.max(err)
        results[n] = approx_max_err
        if do_plot:
            plt.plot(y, py, label=f'p_{n}(x)')

    if do_plot:
        plt.title(title if title else f'Interpolation: a={a}, b={b}')
        plt.xlabel('x')
        plt.ylabel('value')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return results

# Define functions
def f_sin(x): return np.sin(np.pi * x)
def f_rational(x): return 1.0 / (1.0 + x**2)

if __name__ == "__main__":
    # Case 1: sin(pi x) on [-1, 1]
    res1 = run_case(f_sin, -1.0, 1.0, title='f(x) = sin(πx) on [-1,1]')

    # Case 2: 1/(1+x^2) on [-2, 2]
    res2 = run_case(f_rational, -2.0, 2.0, title='f(x) = 1/(1+x^2) on [-2,2]')

    # Case 3: 1/(1+x^2) on [-5, 5]
    res3 = run_case(f_rational, -5.0, 5.0, title='f(x) = 1/(1+x^2) on [-5,5]')

    # Print the 12 approximate maximum interpolation errors
    print("Approximate max errors (M=500):")
    print("sin(πx), [-1,1]:", res1)
    print("1/(1+x^2), [-2,2]:", res2)
    print("1/(1+x^2), [-5,5]:", res3)
