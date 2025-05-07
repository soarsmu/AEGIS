import numpy as np
from skopt import gp_minimize

def monitor_synthesis(param_bounds, env, actor, K, test_episodes):

    def objective(param):
        violations = 0
        overhead = 0
        s = env.reset()
        for j in range(test_episodes):
            a = actor.predict(s.reshape([1, actor.s_dim]))
            a_k = K.dot(s)

            if (np.abs(a - a_k) > param).any():
                a = a_k
                overhead += 1
            s, r, terminal = env.step(a.reshape(env.action_dim, 1))

            if terminal and j < test_episodes:
                    violations += 1

        return np.log(violations+1) + overhead

    # Perform Bayesian optimization
    result = gp_minimize(objective, param_bounds, n_calls=20)  # Adjust the number of function evaluations (n_calls) as desired
    best_param = result.x

    return best_param


def monitor_synthesis_2(env_name, param_bounds, env, actor, K, test_episodes):

    unsafe_trace = []
    while len(unsafe_trace) < 5:
        s = env.reset()
        init_s = s
        for j in range(test_episodes):
            a = actor.predict(s.reshape([1, actor.s_dim]))
            s, r, terminal = env.step(a.reshape(env.action_dim, 1))
            if terminal:
                if np.sum(np.power(env.xk, 2)) > env.terminal_err:
                    unsafe_trace.append(init_s)
                break

    def objective(param):
        violations = 0
        overhead = 0
        real_overhead = 0
        for unsafe_s in unsafe_trace:
            s = env.reset(unsafe_s)
            for j in range(test_episodes):
                a = actor.predict(s.reshape([1, actor.s_dim]))
                a_next = a
                a_k = K.dot(s)

                if (np.abs(a - a_k) > param).any():
                    a = a_k
                    overhead += 1

                    if env_name == "self_driving":
                        f = env.polyf
                    else:
                        def f(x, u):
                            return env.A.dot(x.reshape([env.state_dim, 1])) + env.B.dot(u.reshape([env.action_dim, 1]))

                    s_next = s
                    for i in range(test_episodes - j):
                        if not ((s_next <= env.x_max).all() and (s_next >= env.x_min).all()):
                            real_overhead += 1
                            break

                        if env.continuous:
                            s_next = s_next + env.timestep * (f(s_next, a_next))
                        else:
                            s_next = f(s_next, a_next)

                        a_next = actor.predict(s_next.reshape([1, actor.s_dim]))

                        if not env_name == "self_driving":
                            if (a_next > env.u_max).all():
                                a_next = env.u_max
                            elif (a_next < env.u_min).all():
                                a_next = env.u_min

                s, r, terminal = env.step(a.reshape(env.action_dim, 1))

                if terminal:
                    if np.sum(np.power(env.xk, 2)) > env.terminal_err:
                        violations += 1
                        break

        return np.log(violations+1) - real_overhead/(overhead+1)

    # Perform Bayesian optimization
    result = gp_minimize(objective, param_bounds, n_calls=20, verbose=True)  # Adjust the number of function evaluations (n_calls) as desired
    # print(result)
    best_param = result.x

    return best_param