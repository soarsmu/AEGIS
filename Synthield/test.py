
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
import time
from envs import ENV_CLASSES

from skopt import gp_minimize
import numpy as np

logging.getLogger().setLevel(logging.INFO)

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

    s_init = env.reset()
    while True:
        s_set = []
        a_policy = []
        s = s_init
        for i in range(args.test_episodes):
            s_set.append(np.reshape(np.array(s), (1, actor.s_dim)).squeeze())
            a = actor.predict(np.reshape(np.array(s), (1, actor.s_dim)))
            a_policy.append(a.reshape(actor.a_dim, 1).squeeze())
            s, r, terminal = env.step(a.reshape(actor.a_dim, 1))

        # split data as 8:2
        s_train = s_set[:int(len(s_set)*0.8)]
        s_test = s_set[int(len(s_set)*0.8):]
        a_policy_train = a_policy[:int(len(a_policy)*0.8)]
        a_policy_test = a_policy[int(len(a_policy)*0.8):]

        # print(s_train)
        # s_train = np.array(s_train)
        # s_test = np.array(s_test)
        # a_policy_train = np.array(a_policy_train)
        # a_policy_test = np.array(a_policy_test)
        
        # create polynomial features
        poly_features = PolynomialFeatures(degree=1)

        # create polynomial regression model
        model = make_pipeline(poly_features, LinearRegression())
        model.fit(s_train, a_policy_train)

        # Make predictions
        predictions = model.predict(s_test)

        # Compute Mean Squared Error (MSE)
        mse = mean_squared_error(a_policy_test, predictions)

        # Compute Mean Absolute Error (MAE)
        mae = mean_absolute_error(a_policy_test, predictions)

        print("Predictions:", predictions)
        print("Actual:", a_policy_test)
        print("Mean Squared Error (MSE):", mse)
        print("Mean Absolute Error (MAE):", mae)

        # get the coefficients
        a = model.named_steps['linearregression'].coef_
        b = model.named_steps['linearregression'].intercept_

        s_init = env.reset()
        s = s_init
        flag = True
        for i in range(args.test_episodes):
            a_linear = model.predict(np.reshape(np.array(s), (1, actor.s_dim)))
            s, r, terminal = env.step(a_linear.reshape(actor.a_dim, 1))
            if terminal:
                flag = False
                break
        if flag == True:
            break

#     # Define the objective function to optimize
#     def objective_function(params):

#         a, b, c, d, e, degree = params
#         # Example of a fitness metric, e.g., minimize the sum of squared residuals
#         for s in s_set:
#             a_linear = a * s[0] + b * s[1] + c * s[2] + d * s[3] + e
#             residuals = a_linear - a_policy
#         fitness = np.sum(residuals**2)
#         return fitness


# # Define the search space bounds
# bounds = [(-20, 20), (-20, 20), (-20, 20), (1, 4)]

# # Perform Bayesian optimization
# result = gp_minimize(objective_function, bounds, n_calls=50)

# # Retrieve the optimized constant coefficients and degrees
# optimal_params = result.x
# optimal_fitness = result.fun

# print("Optimized Parameters:", optimal_params)
# print("Optimized Fitness:", optimal_fitness)
