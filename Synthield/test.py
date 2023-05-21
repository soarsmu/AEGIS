import numpy as np
from scipy.linalg import solve_continuous_are
from skopt import gp_minimize

# Define system matrices
A = np.array([[1, 1], [0, 1]])  # State matrix
B = np.array([[0], [1]])  # Control matrix

# Define the cost function
def lqr_cost(params):
    Q = np.array([[params[0], 0], [0, params[1]]])  # State cost
    R = np.array([[params[2]]])  # Control cost
    P = solve_continuous_are(A, B, Q, R)
    return np.trace(P)

# Define the bounds for Q and R
bounds = [(0.1, 10.0), (0.1, 10.0), (0.1, 10.0)]

# Perform Bayesian optimization
result = gp_minimize(lqr_cost, bounds, n_calls=20)

# Optimal values
optimal_values = result.x
optimal_Q = np.array([[optimal_values[0], 0], [0, optimal_values[1]]])
optimal_R = np.array([[optimal_values[2]]])

# Optimal cost
optimal_cost = result.fun

print("Optimal Q: {}".format(optimal_Q))
