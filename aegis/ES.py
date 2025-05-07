import logging
import numpy as np
from tqdm import tqdm

logging.getLogger().setLevel(logging.INFO)


def evolution_policy(
    env,
    policy,
    K,
    n_states,
    n_actions,
    len_episodes,
    n_population=10,
    n_iterations=5,
    sigma=0.1,
    alpha=0.05,
):

    # coffset = np.random.randn(n_actions, n_states)
    coffset = K

    for iter in tqdm(range(n_iterations)):
        noise = np.random.randn(int(n_population / 2), n_actions, n_states)
        noise = np.vstack((noise, -noise))
        distance = np.zeros(n_population)

        s = env.reset()
        for i in range(len_episodes):
            a_policy = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
            for p in range(n_population):
                new_coffset = coffset + sigma * noise[p]
                a_linear = np.array(new_coffset.dot(s)).reshape(1, policy.a_dim)
                distance[p] = -np.sum((a_policy - a_linear).squeeze() ** 2)
            std_distance = (distance - np.mean(distance)) / np.std(distance)
            coffset = (
                coffset
                + (alpha / (n_population * sigma) * np.dot(noise.T, std_distance)).T
            )
            s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))
    return coffset


def policy_distance(env, policy, n_states, coffset, len_episodes):
    s = env.reset()
    distance = 0
    for i in range(len_episodes):
        a_policy = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
        a_linear = np.array(
            coffset[:, : n_states - 1].dot(s) + coffset[:, n_states - 1 :]
        ).reshape(1, policy.a_dim)
        distance -= np.sum((a_policy - a_linear).squeeze() ** 2)
        s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))

    return abs(distance / len_episodes)


def refine(
    env,
    policy,
    coffset,
    n_states,
    n_actions,
    state,
    len_episodes,
    n_population=50,
    sigma=0.1,
    alpha=0.001,
):

    noise = np.random.randn(int(n_population / 2), n_actions, n_states)
    noise = np.vstack((noise, -noise))
    distance = np.zeros(n_population)

    s = env.reset(np.reshape(np.array(state), (policy.s_dim, 1)))
    for i in tqdm(range(len_episodes)):
        a_policy = policy.predict(np.reshape(np.array(s), (1, policy.s_dim)))
        for p in range(n_population):
            new_coffset = coffset + sigma * noise[p]
            a_linear = np.array(
                new_coffset[:, : n_states - 1].dot(s) + new_coffset[:, n_states - 1 :]
            ).reshape(1, policy.a_dim)
            distance[p] = -np.sum((a_policy - a_linear).squeeze() ** 2)
        std_distance = (distance - np.mean(distance)) / np.std(distance)
        coffset = (
            coffset + (alpha / (n_population * sigma) * np.dot(noise.T, std_distance)).T
        )
        s, r, terminal = env.step(a_policy.reshape(policy.a_dim, 1))

    return coffset
