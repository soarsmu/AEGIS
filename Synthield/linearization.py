import numpy as np

def compute_jacobians(f, x_eq, u_eq):
    # Set a small perturbation for numerical differentiation
    epsilon = 1e-8

    # Initialize matrices A and B with zeros
    ds = len(x_eq)
    us = len(u_eq)
    A = np.zeros((ds, ds))
    B = np.zeros((ds, us))

    # Compute Jacobian matrix A
    for i in range(ds):
        perturbation = np.zeros((ds, 1))
        perturbation[i, 0] = epsilon
        A[:, i] = ((f(x_eq + perturbation, u_eq) - f(x_eq - perturbation, u_eq)) / (2 * epsilon)).flatten()

    # Compute Jacobian matrix B
    for i in range(us):
        perturbation = np.zeros((us, 1))
        perturbation[i, 0] = epsilon
        B[:, i] = ((f(x_eq, u_eq + perturbation) - f(x_eq, u_eq - perturbation)) / (2 * epsilon)).flatten()

    return A, B

# Equilibrium point: Car stationary, Pole upright
x_eq = np.array([[0], [0]])
u_eq = np.array([[0]])

# System dynamics definition
def f(x, u):
    delta = np.zeros((2, 1), float)
    delta[0, 0] = -2 * (x[1, 0] - ((x[1, 0] ** 3) / 6))
    delta[1, 0] = u[0, 0]
    return delta

# Compute Jacobian matrices A and B
A, B = compute_jacobians(f, x_eq, u_eq)

# Display the results
print("Jacobian Matrix A:")
print(A)
print("\nJacobian Matrix B:")
print(B)
