import json
import logging
import argparse
import numpy as np
from tqdm import tqdm
from DDPG import DDPG

import time
from metrics import (
    neural_network_performance,
    linear_function_performance,
    combo_function_performance,
)

from envs import ENV_CLASSES
import numpy as np
from monitor import monitor_synthesis, monitor_synthesis

from pympc.geometry.polyhedron import Polyhedron
from pympc.dynamics.discrete_time_systems import LinearSystem
from ES import refine
from linearization import compute_jacobians

logging.getLogger().setLevel(logging.INFO)


def synthesis(env, actor, test_episodes):

    S0 = Polyhedron.from_bounds(env.s_min, env.s_max)
    if env.continuous:
        Sys = LinearSystem.from_continuous(
            np.asarray(env.A), np.asarray(env.B), env.timestep
        )
    else:
        Sys = LinearSystem(np.asarray(env.A), np.asarray(env.B))

    _, K = Sys.solve_dare(env.Q, env.R)
    logging.info(f"Initial Controller: {K}")
    K = np.asarray(K)
    K = finetune(env, K, actor, 5)

    O_inf = inf_synthesis(env, Sys, K)

    ce = S0.is_included_in_with_ce(O_inf)

    while ce is not None:
        K_new = refine(env, actor, K, ce, test_episodes)
        K_new = np.asarray(K_new)
        if not (K == K_new).all():
            K = K_new
            O_inf = inf_synthesis(env, Sys, K)
            ce = S0.is_included_in_with_ce(O_inf)
        else:
            ce = None

    return O_inf, K


def inf_synthesis(env, Sys, K):
    X = Polyhedron.from_bounds(env.x_min, env.x_max)
    U = Polyhedron.from_bounds(env.u_min, env.u_max)
    D = X.cartesian_product(U)
    O_inf = Sys.mcais(K, D)
    return O_inf


def spurious_check(env, K, ce, test_episodes):
    flag = False

    s = env.reset(np.array(ce).reshape([-1, 1]))
    for i in range(test_episodes):
        a = K.dot(s)
        s, r, terminal = env.step(a.reshape(env.action_dim, 1))
        if terminal and i < test_episodes - 1:
            flag = True
            break

    if flag:
        return ce

    return None


def refine(env, actor, K, ce, test_episodes):
    epsilon = 1e-5
    learning_rate = 1e-3

    ce = spurious_check(env, K, ce, test_episodes)
    if ce is None:
        return K

    s = env.reset(np.array(ce).reshape([-1, 1]))
    for _ in range(test_episodes):
        a = actor.predict(np.reshape(np.array(s), (1, actor.s_dim)))

        perturbation = np.zeros_like(K)
        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                perturbation[i, j] = epsilon

        K_plus = K + perturbation
        K_minus = K - perturbation

        a_plus = K_plus.dot(s)
        a_minus = K_minus.dot(s)

        s, r, terminal = env.step(a.reshape(actor.a_dim, 1))

        distance_plus = -np.sum((a_plus - a).squeeze() ** 2) + env.reward(s, a_plus)
        distance_minus = -np.sum((a_minus - a).squeeze() ** 2) + env.reward(s, a_minus)

        gradients = (distance_plus - distance_minus) / (2 * epsilon)
        K -= learning_rate * gradients

    return K


def finetune(env, K, actor, test_episodes):
    epsilon = 1e-5
    learning_rate = 1e-3
    s = env.reset()
    for _ in range(test_episodes):
        a = actor.predict(np.reshape(np.array(s), (1, actor.s_dim)))

        perturbation = np.zeros_like(K)
        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                perturbation[i, j] = epsilon

        K_plus = K + perturbation
        K_minus = K - perturbation

        a_plus = K_plus.dot(s)
        a_minus = K_minus.dot(s)

        s, r, terminal = env.step(a.reshape(actor.a_dim, 1))

        distance_plus = -np.sum((a_plus - a).squeeze() ** 2)
        distance_minus = -np.sum((a_minus - a).squeeze() ** 2)

        gradients = (distance_plus - distance_minus) / (2 * epsilon)
        K -= learning_rate * gradients

    return K


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Running Options")
    parser.add_argument(
        "--env", default="cartpole", type=str, help="The selected environment."
    )
    parser.add_argument("--do_eval", action="store_true", help="Test RL controller")
    parser.add_argument("--test_episodes", default=5000, help="test_episodes", type=int)
    parser.add_argument("--rounds", default=100, help="rounds", type=int)
    parser.add_argument(
        "--do_retrain", action="store_true", help="retrain RL controller"
    )
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

    x_eq = np.zeros([env.state_dim, 1])
    u_eq = np.zeros([env.action_dim, 1])

    if env.x_min is None:
        env.A, env.B = compute_jacobians(env.polyf, x_eq, u_eq)
        logging.info(f"Environment Dynamics: {env.A} {env.B}")
        env.x_min = np.array(DDPG_args["safe_spec"][0]).reshape([-1, 1])
        env.x_max = np.array(DDPG_args["safe_spec"][1]).reshape([-1, 1])
    else:
        def f(x, u):
            return env.A.dot(x) + env.B.dot(u)
        apprx_A, apprx_B = compute_jacobians(f, x_eq, u_eq)
        logging.info(f"Environment Dynamics: {apprx_A} {apprx_B}")
        env.A = apprx_A
        env.B = apprx_B

    stability_threshold = DDPG_args["stability_threshold"]

    if args.env == "car_platoon_4" or args.env == "car_platoon_8":
        param_bounds = [tuple(param) for param in DDPG_args["param_bounds"]]
    else:
        param_bounds = [tuple(DDPG_args["param_bounds"])]

    actor = DDPG(env, DDPG_args)

    logging.info("Synthesizing Controller & Invariant...")
    start = time.time()

    O_inf, K = synthesis(env, actor, args.test_episodes)

    logging.info("Sythesized Controller: {}".format(K))
    logging.info("Sythesized Invariant (left): {}".format(O_inf.A))
    logging.info("Sythesized Invariant (right): {}".format(O_inf.b))

    logging.info("Testing Stability...")
    stability_rl, steps_rl = neural_network_performance(env, actor, stability_threshold)
    stability_K, steps_K = linear_function_performance(env, K, stability_threshold)

    logging.info(
        "Reinforcement Learning Policy's Steps to Stability: {}".format(stability_rl)
    )
    logging.info("Linear Function Shield's Steps to Stability: {}".format(stability_K))

    logging.info("Sythesizing Monitor...")
    monitor_params = monitor_synthesis(args.env, param_bounds, env, actor, K, 200)

    stability_combo, steps_combo = combo_function_performance(
        args, env, actor, K, monitor_params, stability_threshold
    )

    logging.info(
        "Combo Function Shield's Steps to Stability: {}".format(stability_combo)
    )
    syn_time = time.time() - start

    logging.info("Monitoring Parameter: {}".format(monitor_params))

    def check_necessary_condition(args, env, actor, a, rest_time):
        s = env.xk

        if args.env == "self_driving":
            f = env.polyf
        else:

            def f(x, u):
                return env.A.dot(x.reshape([env.state_dim, 1])) + env.B.dot(
                    u.reshape([env.action_dim, 1])
                )

        for i in range(rest_time):
            if not ((s <= env.x_max).all() and (s >= env.x_min).all()):
                return True

            if env.continuous:
                s = s + env.timestep * (f(s, a))
            else:
                s = f(s, a)

            a = actor.predict(np.reshape(np.array(s), (1, actor.s_dim)))

            if not args.env == "self_driving":
                if (a > env.u_max).all():
                    a = env.u_max
                elif (a < env.u_min).all():
                    a = env.u_min

        return False

    def shield_policy():

        volations = 0
        all_time = 0
        total_overheads = 0
        total_calls = 0
        total_rewards = 0
        real_calls = 0
        for i in tqdm(range(args.rounds)):
            sys_time = time.time()
            s = env.reset()
            time_overhead = 0
            calls = 0

            for i in range(args.test_episodes):

                a = actor.predict(s.reshape([1, actor.s_dim]))
                start = time.time()
                ncp_a = a

                a_k = K.dot(s)

                if np.abs(a - a_k) > monitor_params:
                    a = a_k
                    calls += 1
                    if check_necessary_condition(
                        args, env, actor, ncp_a, args.test_episodes - i
                    ):
                        real_calls += 1

                time_overhead += time.time() - start
                s, r, terminal = env.step(a.reshape(actor.a_dim, 1))
                total_rewards += r
                if terminal:
                    if np.sum(np.power(env.xk, 2)) > env.terminal_err:
                        volations += 1
                    break
            total_overheads += time_overhead
            total_calls += calls
            all_time += time.time() - sys_time

        logging.info(f"Total rewards: {total_rewards}")
        return volations, all_time, total_overheads, total_calls, real_calls

    volations, all_time, total_overheads, total_calls, real_calls = shield_policy()

    logging.info("Synthesis time: {}".format(syn_time))
    logging.info("System Time: {}".format(all_time))
    logging.info("Overhead: {}".format(total_overheads))
    logging.info("Overhead Rate: {}".format(total_overheads / all_time))
    logging.info("Total calls: {}".format(total_calls))
    logging.info("Violations: {}".format(volations))
    logging.info("Real calls: {}".format(real_calls))
