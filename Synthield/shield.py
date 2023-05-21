
import json
import logging
import argparse
import numpy as np
from tqdm import tqdm
from DDPG import DDPG
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.linalg import solve_continuous_are
import time
from envs import ENV_CLASSES
import numpy as np
from monitor import monitor_synthesis
import scipy
from pympc.geometry.polyhedron import Polyhedron
from pympc.dynamics.discrete_time_systems import LinearSystem
from skopt import gp_minimize
from skopt.space import Real
from pympc.control.controllers import ModelPredictiveController
from z3verify import bound_z3
from skopt import gp_minimize
from sklearn.linear_model import LinearRegression

logging.getLogger().setLevel(logging.INFO)

def refine(env, K, ce, test_episodes):
    epsilon = 1e-4
    learning_rate = 1e-3
    s = env.reset(np.array(ce).reshape([-1, 1]))

    for _ in range(test_episodes):
        a = K.dot(s)

        perturbation = np.zeros_like(K)
        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                perturbation[i, j] = epsilon

        K_plus = K + perturbation
        K_minus = K - perturbation

        a_plus = K_plus.dot(s)
        a_minus = K_minus.dot(s)

        s, r, terminal = env.step(a.reshape(actor.a_dim, 1))
        gradients = (env.reward(s, a_plus) - env.reward(s, a_minus)) / (2 * epsilon)
        K -= learning_rate * gradients
    return K


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Running Options")
    parser.add_argument("--env", default="cartpole", type=str, help="The selected environment.")
    parser.add_argument("--do_eval", action="store_true", help="Test RL controller")
    parser.add_argument("--test_episodes", default=5000, help="test_episodes", type=int)
    parser.add_argument("--do_retrain", action="store_true", help="retrain RL controller")
    args = parser.parse_args()

    env = ENV_CLASSES[args.env]()
    with open("configs.json") as f:
        configs = json.load(f)

    DDPG_args = configs[args.env]
    DDPG_args["enable_retrain"] = args.do_retrain
    DDPG_args["enable_eval"] = args.do_eval
    DDPG_args["enable_fuzzing"] = False
    DDPG_args["enable_falsification"] = False

    DDPG_args["test_episodes"] = args.test_episodes
    actor = DDPG(env, DDPG_args)    
    O_inf_list = []
    K_list = []

    S0 = Polyhedron.from_bounds(env.s_min, env.s_max)
    if env.continuous:
        Sys = LinearSystem.from_continuous(np.asarray(env.A), np.asarray(env.B), env.timestep)
    else:
        Sys = LinearSystem(np.asarray(env.A), np.asarray(env.B))
    P, K = Sys.solve_dare(env.Q, env.R)
    K = np.asarray(K)
    X = Polyhedron.from_bounds(env.x_min, env.x_max)
    U = Polyhedron.from_bounds(env.u_min, env.u_max)
    D = X.cartesian_product(U)
    O_inf = Sys.mcais(K, D)
    O_inf_list.append(O_inf)
    K_list.append(K)
    
    ce = S0.is_included_in_with_ce(O_inf)
    print(K)
    if ce is not None:
        flag = True
        s = env.reset(np.array(ce).reshape([-1, 1]))
        for i in range(args.test_episodes):
            a = K.dot(s)
            s, r, terminal = env.step(a.reshape(actor.a_dim, 1))
            if terminal and i < args.test_episodes - 1:
                flag = False
                print("terminal at {}".format(i))
                print(s)
                break
        if flag:
            print("ce is not a counterexample")
    # exit()        

    from skopt import gp_minimize

    def objective(param):
        violations = 0
        s = env.reset()
        shield_times = 0
        for j in range(args.test_episodes):
            a = actor.predict(s.reshape([1, actor.s_dim]))
            a_k = K.dot(s)
            if np.abs(a_k - a) > param:
                # s_ = env.simulation(a.reshape(actor.a_dim, 1))
                # if not ((s_ <= env.x_max).all() and (s_ >= env.x_min).all()):
                    # real += 1
                    shield_times += 1
                    a = a_k
            s, r, terminal = env.step(a.reshape(actor.a_dim, 1))

            if terminal and j < args.test_episodes:
                violations += 1
        return violations

    # Define the parameter bounds for optimization
    param_bounds = [(0.0, 20.0)]  # Adjust the bounds according to your problem

    # Perform Bayesian optimization
    result = gp_minimize(objective, param_bounds, n_calls=10)  # Adjust the number of function evaluations (n_calls) as desired

    # Get the best parameter value
    print("result", result)
    best_param = result.x[0]
    print("best_param", best_param)

    real = 0
    # model = monitor_synthesis(actor, env, K)
    volations = 0
    for i in range(100):
        shield_times = 0
        s = env.reset()
        for i in tqdm(range(args.test_episodes)):
            a = actor.predict(s.reshape([1, actor.s_dim]))
            a_k = K.dot(s)
            if np.abs(a_k - a) > best_param:
                # s_ = env.simulation(a.reshape(actor.a_dim, 1))
                # if not ((s_ <= env.x_max).all() and (s_ >= env.x_min).all()):
                    # real += 1
                    shield_times += 1
                    a = a_k
            s, r, terminal = env.step(a.reshape(actor.a_dim, 1))

            # a = K.dot(s)
            # s, r, terminal = env.step(a.reshape(actor.a_dim, 1))
            if terminal and i < args.test_episodes - 1:
                volations += 1
                break
        print("shield_times", shield_times)
        print("real", real)
        
