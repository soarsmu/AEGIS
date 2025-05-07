import numpy as np

from .environment import Environment

def cartpole_var():
    l = .22+0.15 # rod length is 2l
    m = (2*l)*(.006**2)*(3.14/4)*(7856) # rod 6 mm diameter, 44cm length, 7856 kg/m^3
    M = .4
    dt = .02 # 20 ms
    g = 9.8

    A = np.matrix([[1, dt, 0, 0],[0,1, -(3*m*g*dt)/(7*M+4*m),0],[0,0,1,dt],[0,0,(3*g*(m+M)*dt)/(l*(7*M+4*m)),1]])
    B = np.matrix([[0],[7*dt/(7*M+4*m)],[0],[-3*dt/(l*(7*M+4*m))]])

    #intial state space
    s_min = np.array([[ -0.1],[ -0.1], [-0.05], [ -0.05]])
    s_max = np.array([[  0.1],[  0.1], [ 0.05], [  0.05]])

    Q = np.matrix("1 0 0 0; 0 1 0 0 ; 0 0 1 0; 0 0 0 1")
    R = np.matrix(".0005")

    x_min = np.array([[-0.3],[-0.5],[-0.3],[-0.5]])
    x_max = np.array([[ .3],[ .5],[.3],[.5]])
    u_min = np.array([[-15.]])
    u_max = np.array([[ 15.]])
    env = Environment(A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R)

    return env