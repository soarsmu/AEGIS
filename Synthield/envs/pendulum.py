import numpy as np

from .environment import Environment

def pendulum():
    # State transform matrix
    A = np.matrix([[1.9027, -1],
                        [1, 0]
                        ])

    B = np.matrix([[1],
                        [0]
                        ])

    # initial action space
    u_min = np.array([[-1.]])
    u_max = np.array([[1.]])

    # intial state space
    s_min = np.array([[-0.5],[-0.5]])
    s_max = np.array([[ 0.5],[0.5]])
    x_min = np.array([[-0.6], [-0.6]])
    x_max = np.array([[0.6], [0.6]])

    # coefficient of reward function
    Q = np.matrix("1 0 ; 0 1")
    R = np.matrix(".0005")

    env = Environment(A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R)

    return env
