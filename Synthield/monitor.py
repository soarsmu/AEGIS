import time
import json
import argparse
import numpy as np
from sklearn import svm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from DDPG import DDPG
from envs import ENV_CLASSES
from ES import evolution_policy

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

    # coffset = evolution_policy(env, actor, 5, 250)

    safe_set = []
    unsafe_set = []
    while len(safe_set) < 100 or len(unsafe_set) < 100:
        s = env.reset()
        for i in range(500):
            a = actor.predict(np.reshape(np.array(s), (1, actor.s_dim)))
            s, r, terminal = env.step(a.reshape(actor.a_dim, 1))
            if terminal and i < 499:
                unsafe_set.append(np.reshape(np.array(s), (1, actor.s_dim)).squeeze())
                break
            else:
                safe_set.append(np.reshape(np.array(s), (1, actor.s_dim)).squeeze())
    print(len(safe_set), len(unsafe_set))
    safe_set = safe_set[:100]
    safe_set_train = safe_set[:80]
    safe_set_test = safe_set[80:]
    unsafe_set_train = unsafe_set[:80]
    unsafe_set_test = unsafe_set[80:]
    safe_set_train = np.array(safe_set_train)
    safe_set_test = np.array(safe_set_test)
    unsafe_set_train = np.array(unsafe_set_train)
    unsafe_set_test = np.array(unsafe_set_test)
    train_set = np.concatenate((safe_set_train, unsafe_set_train), axis=0)
    test_set = np.concatenate((safe_set_test, unsafe_set_test), axis=0)
    train_label = np.concatenate((np.ones(len(safe_set_train)), np.zeros(len(unsafe_set_train))), axis=0)
    test_label = np.concatenate((np.ones(len(safe_set_test)), np.zeros(len(unsafe_set_test))), axis=0)

    # Shuffle the train set and train label in the same order
    p = np.random.permutation(len(train_set))
    train_set = train_set[p]
    train_label = train_label[p]

    # clf = svm.SVC(kernel='linear')
    # print(train_set.shape, train_label.shape)
    # clf.fit(train_set, train_label)

    degree = 2  # Degree of the polynomial features
    poly_features = PolynomialFeatures(degree)
    model = make_pipeline(poly_features, LogisticRegression())

    # Fit the model to the data
    model.fit(train_set, train_label)
    print(model.score(test_set, test_label))
    # Get the feature names
    feature_names = poly_features.get_feature_names()

    # Print the feature names
    print("Polynomial Features:", feature_names)
    # Retrieve the coefficients
    coefficients = model.named_steps['logisticregression'].coef_

    print("Coefficients:", coefficients)
    # print(clf.score(test_set, test_label))
    # print(clf.predict(safe_set_test))
    # print(clf.predict(unsafe_set_test))

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





