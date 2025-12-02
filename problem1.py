import numpy as np

# Gauss nodes and weights on [-1,1]
nodes = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
weights = np.array([5/9, 8/9, 5/9])

def composite_gauss(f, N):
    h = 1.0 / N
    total = 0.0
    for j in range(N):
        m = (j + 0.5) * h
        pts = m + 0.5*h*nodes
        total += np.sum(weights * f(pts))
    return 0.5*h * total

f = lambda x: np.sin(np.pi * x)
I_exact = 2.0 / np.pi

for N in [2,4,8,16]:
    I_approx = composite_gauss(f, N)
    err = abs(I_approx - I_exact)
    print(f"N={N:2d}, I≈{I_approx:.12f}, error={err:.3e}")

# Find minimal N with error < 1e-7
N = 1
while True:
    err = abs(composite_gauss(f, N) - I_exact)
    if err < 1e-7:
        print("Minimal N with error < 1e-7:", N)
        break
    N += 1
