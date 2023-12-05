
import json
import logging
import argparse
import random
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
from sklearn.linear_model import LinearRegression
from ES import evolution_policy
from linearization import compute_jacobians

logging.getLogger().setLevel(logging.INFO)

def refine(env, K, ce, test_episodes):
    epsilon = 1e-9
    learning_rate = 30
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

    start = time.time()
    # x_eq = np.array([[0], [0]])
    # u_eq = np.array([[0]])
    # A, B = compute_jacobians(env.polyf, x_eq, u_eq)

    S0 = Polyhedron.from_bounds(env.s_min, env.s_max)
    if env.continuous:
        Sys = LinearSystem.from_continuous(np.asarray(env.A), np.asarray(env.B), env.timestep)
    else:
        Sys = LinearSystem(np.asarray(env.A), np.asarray(env.B))
    P, K = Sys.solve_dare(env.Q, env.R)
    K = np.asarray(K)
    n_states = actor.s_dim
    n_actions = actor.a_dim
    syn_policy = evolution_policy(env, actor, n_states, n_actions, 100)
    print(syn_policy, K)

    # exit()

    # print(K.dot(np.array([0.0, 0.0, 0.0, 0.0]).reshape([-1, 1])))
    # s = env.reset(np.array([0.0, 0.0, 0.0, 0.0]).reshape([-1, 1]))
    # s  = env.reset(np.array([0.5, 0.5]).reshape([-1, 1]))
    for i in range(10):
        s = env.reset()
        print(s)
        for i in range(5000):
            a = K.dot(s)
            s, r, terminal = env.step(a.reshape(actor.a_dim, 1))
            if terminal and i < args.test_episodes - 1:
                flag = False
                print("terminal at {}".format(i))
                print("terminal at {}".format(i), s)
                break
    # try:
    #     print("ce", bound_z3(syn_policy, env.A, env.B, None, (env.s_min, env.s_max), (env.x_min, env.x_max), 12))
    # except:
    #     print("ce", bound_z3(syn_policy, None, None, env.polyf, (np.array(DDPG_args["initial_conditions"][0]).reshape([-1, 1]), np.array(DDPG_args["initial_conditions"][1]).reshape([-1, 1])), (np.array(DDPG_args["safe_spec"][0]).reshape([-1, 1]), np.array(DDPG_args["safe_spec"][1]).reshape([-1, 1])), 4))
    # exit()

    # n_states = actor.s_dim
    # n_actions = actor.a_dim
    # syn_policy = evolution_policy(env, actor, n_states, n_actions, 100)
    # print(syn_policy, K)
    X = Polyhedron.from_bounds(env.x_min, env.x_max)
    U = Polyhedron.from_bounds(env.u_min, env.u_max)
    D = X.cartesian_product(U)
    O_inf = Sys.mcais(syn_policy, D)
    O_inf_list.append(O_inf)
    K_list.append(K)

    print(O_inf.intersection(S0).A, O_inf.intersection(S0).b)
    O_inf.plot()
    ce = S0.is_included_in_with_ce(O_inf)
    print(ce)
    exit()
    from metrics import neural_network_performance, linear_function_performance
    print(neural_network_performance(env, actor))
    print(linear_function_performance(env, syn_policy))
    K = syn_policy
    # exit()
    # while ce is not None:
    #     # flag = True
    #     K = refine(env, K, ce, args.test_episodes)
    #     K = np.asarray(K)
    #     print(K)
    #     X = Polyhedron.from_bounds(env.x_min, env.x_max)
    #     U = Polyhedron.from_bounds(env.u_min, env.u_max)
    #     D = X.cartesian_product(U)
    #     O_inf = Sys.mcais(K, D)
    #     ce = S0.is_included_in_with_ce(O_inf)
    #     print(ce)
    #     s = env.reset(np.array(ce).reshape([-1, 1]))
    #     for i in range(args.test_episodes):
    #         a = K.dot(s)
    #         s, r, terminal = env.step(a.reshape(actor.a_dim, 1))
    #         if terminal and i < args.test_episodes - 1:
    #             flag = False
    #             print("terminal at {}".format(i))
    #             print(s)
    #             break
        # if flag:
        #     break

    from skopt import gp_minimize

    def objective(param):
        violations = 0
        overhead = 0
        s = env.reset()
        for j in range(args.test_episodes):
            a = actor.predict(s.reshape([1, actor.s_dim]))
            # start = time.time()
            a_k = K.dot(s)
            # if param[:4] * s + param[-1] > 0:
            if np.abs(a - a_k) > param:
                a = a_k
                # overhead += time.time() - start
                overhead += 1
            s, r, terminal = env.step(a.reshape(actor.a_dim, 1))

            if terminal and j < args.test_episodes:
                    violations += 1
        # print(np.log(violations+1) + overhead, overhead)
        # return np.log(violations+1) + overhead
        print(violations + overhead, overhead)
        return violations + overhead

    # Define the parameter bounds for optimization
    param_bounds = [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)]  # Adjust the bounds according to your problem
    param_bounds = [(-10.0, 10.0)]

    # Perform Bayesian optimization
    result = gp_minimize(objective, param_bounds, n_calls=20)  # Adjust the number of function evaluations (n_calls) as desired
    print(result)
    best_param = result.x
    print(best_param)
    syn_time = time.time() - start

    real = 0
    volations = 0
    all_time = 0
    total_overheads = 0
    total_calls = 0
    for i in tqdm(range(10)):
        sys_time = time.time()
        s = env.reset()
        time_overhead = 0
        calls = 0
        for i in range(args.test_episodes):
            a = actor.predict(s.reshape([1, actor.s_dim]))
            start = time.time()
            a_k = K.dot(s)
            if np.abs(a - a_k) > best_param:
                a = a_k
                calls += 1
            time_overhead += time.time() - start
            s, r, terminal = env.step(a.reshape(actor.a_dim, 1))

            if terminal and i < args.test_episodes - 1:
                # print(((s <= env.x_max).all() and (s >= env.x_min).all()))
                # print(r == env.bad_reward)
                # print(s)
                # print("terminal at {}".format(i))
                volations += 1
                break
        total_overheads += time_overhead
        total_calls += calls
        all_time += time.time() - sys_time

    print("syn_time:", syn_time)
    print("time:", all_time)
    print("overhead:", total_overheads)
    print("rate:", total_overheads / all_time)
    print("total calls:", total_calls)
    print("violations:", volations)

