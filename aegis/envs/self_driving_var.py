import numpy as np

from .environment import Environment

# Show that there is an invariant that can prove the policy safe
def self_driving_var():
  A = np.matrix([
    [ 0., 0., 1., 0.],
    [ 0., 0., 0., 1.],
    [ 0., 0., -1.2, .1],
    [ 0., 0., .1, -1.2]
    ])
  B = np.matrix([
    [0,0],[0,0],[1,0],[0,1]
    ])

  #intial state space
  s_min = np.array([[-7],[-7],[0],[0]])
  s_max = np.array([[-6],[-8],[0],[0]])

  u_min = np.array([[-1], [-1]])
  u_max = np.array([[ 1], [ 1]])

  x_min = np.array([[-3.1], [-3.1], [-1e10], [-1e10]])
  x_max = np.array([[-2.9], [-2.9], [1e10], [1e10]])

  d, p = B.shape

  Q = np.zeros((d, d), float)
  np.fill_diagonal(Q, 1)

  R = np.zeros((p,p), float)
  np.fill_diagonal(R, .005)

  env = Environment(A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R, continuous=True, unsafe=True, bad_reward=-100)
  return env