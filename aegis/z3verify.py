from z3 import *
import numpy as np
import logging

logging.getLogger().setLevel(logging.INFO)

def d2tod1(space):
	(x_min, x_max) = space
	return [(x_min[i][0], x_max[i][0]) for i in range(len(x_min))]

def get_counterexample(s, x, x_dim, num_steps):
	# A counterexample is found
	m = s.model()
	print(m)
	for j in range(x_dim):
		print(m[x[0][j]])

	trajectory = [[(m[x[i][j]].numerator_as_long())*1.0/(m[x[i][j]].denominator_as_long()) for j in range(x_dim)] for i in range(num_steps)]

	return trajectory

def bound_z3(K, A, B, polyf, init_range, safe_range, test_episodes=1):
	init_range = d2tod1(init_range)
	safe_range = d2tod1(safe_range)

	x_dim = len(init_range)
	x = [[Real("x_ref_%s[%s]" %(j, i)) for i in range(x_dim)] for j in range(test_episodes)]

	s = Solver()
	# print(polynomial_expr)
	# state = [(A.dot(np.array(x_).reshape([-1, 1])) + B.dot(np.array(polynomial_expr_).reshape([-1, 1]))).reshape([1, -1]).tolist()[0] for x_, polynomial_expr_ in zip(x[1:], polynomial_expr)]

	for i in range(x_dim):
		s.add(x[0][i] <= init_range[i][1])
		s.add(x[0][i] >= init_range[i][0])

	for j in range(1, test_episodes):
		constraints = []
		if A is not None:
			constraints.append(x[j][0] == (A.dot(x[j-1]) + B.dot(K.dot(x[j-1]))).tolist()[0][0])
			constraints.append(x[j][1] == (A.dot(x[j-1]) + B.dot(K.dot(x[j-1]))).tolist()[0][1])
			# print((A.dot(x[j-1]) + B.dot(polynomial_expr)).tolist()[0])
			# print(x[j][0] == (A.dot(x[j-1]) + B.dot(polynomial_expr)).tolist()[0][0])
			# print(x[j][1] == (A.dot(x[j-1]) + B.dot(polynomial_expr)).tolist()[0][1])
			# exit()
		else:
			constraints.append(x[j][0] == x[j-1][0] + (-2.0 * (x[j-1][1] - ((x[j-1][1]*x[j-1][1]*x[j-1][1])/6.0)))*0.1)
			constraints.append(x[j][1] == x[j-1][1] + K.dot(x[j-1])[0]*0.1)

		s.add(constraints)
		safety_constraints = []

		# for i in range(x_dim):
		safety_constraints.append(Or(x[j][0] < safe_range[0][0], x[j][0] > safe_range[0][1], x[j][1] < safe_range[1][0], x[j][1] > safe_range[1][1]))
		# print(Or(x[j][0] < safe_range[0][0], x[j][0] > safe_range[0][1], x[j][1] < safe_range[1][0], x[j][1] > safe_range[1][1]))

		s.push()
		s.add(safety_constraints)
		print(s)
		print("====================================")
	# print(s)
		res = s.check()
		if (res == sat):
			m = s.model()
			counterexample = [(m[x[0][k]].numerator_as_long())*1.0/(m[x[0][k]].denominator_as_long()) for k in range(x_dim)]
			s.pop()
			return counterexample
			# return ce
		elif (res == unsat):
			s.pop()
			# logging.info(" The properties hold for the polynomial model.")
			# return None
	return None