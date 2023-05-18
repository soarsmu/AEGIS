# Note some functions are Deprecated! We leave them here for future reference as they are useful.

# from z3 import *
# set_option(max_args=10000000, max_lines=100000000, max_depth=10000000, max_visited=100000000, precision=2)

import json
import logging
import argparse
import numpy as np
from tqdm import tqdm
from DDPG import DDPG
import time
from envs import ENV_CLASSES

logging.getLogger().setLevel(logging.INFO)

def evolution_policy(env, policy, n_vars, len_episodes, n_population=50, n_iterations=50, sigma=0.1, alpha=0.05):

    coffset = np.random.randn(n_vars)

    for iter in tqdm(range(n_iterations)):
        noise = np.random.randn(int(n_population/2), n_vars)
        noise = np.vstack((noise, -noise))
        distance = np.zeros(n_population)

        s = env.reset()
        for i in range(len_episodes):
            a_policy = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
            for p in range(n_population):
                new_coffset = coffset + sigma * noise[p]
                a_linear = new_coffset[:n_vars-1].dot(s)+ new_coffset[n_vars-1]
                distance[p] = - np.abs(a_policy - a_linear)
            std_distance = (distance - np.mean(distance)) / np.std(distance)
            coffset = coffset + alpha / (n_population * sigma) * np.dot(noise.T, std_distance)
            s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))
    # print(policy_distance(env, policy, n_vars, coffset, len_episodes))
    return coffset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running Options")
    parser.add_argument("--env", default="cartpole", type=str, help="The selected environment.")
    parser.add_argument("--do_eval", action="store_true", help="Test RL controller")
    parser.add_argument("--test_episodes", default=50, help="test_episodes", type=int)
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

    coffset = evolution_policy(env, actor, 5, 250)

    safe_set = []
    unsafe_set = []
    def system_with_linear(env, coffset, len_episodes):
        s = env.reset()
        for i in range(len_episodes):
            a_linear = coffset[:5-1].dot(s)+ coffset[5-1]
            s, r, terminal = env.step(a_linear.reshape(actor.a_dim, 1))
            if terminal:
                break
        return s

    actor.sess.close()
    # s = env.reset()
    # for i in tqdm(range(250)):
    #     # a_linear = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
    #     a_linear = coffset[:5-1].dot(s**2)+ coffset[5-1]
    #     s, r, terminal = env.step(a_linear.reshape(policy.a_dim, 1))
    #     if terminal:
    #         break
    # # print(evolution_dynamics(env, 0, policy, 3, 250))
    # # print(evolution_dynamic(env, 0, policy, 4, 250))
    # policy.sess.close()





