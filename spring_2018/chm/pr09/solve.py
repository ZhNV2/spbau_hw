import matplotlib.pyplot as plt
import math
import random
import numpy as np
import sys

def terminate(msg):
	print('program finished with error:')
	print(msg)
	exit()


def f_1(x, y):
	return y / (x + 2) + x ** 2 + 2 * x

def y_t_1(x):
	return (1 + 0.5 * x ** 2) * (x + 2)

def f_2(x, pr):
	y, t = pr[0], pr[1]
	return np.array([t, (-(2 - x) * t - 2 * y - x ** 0.5) / (2 * x * (x + 2))])

def y_t_2(x):
	return -x ** 0.5

def evaluate_y(a, b, n, y0, f):
	Y = [0] * (n + 1)
	Y[0] = y0
	h = (b - a) / n
	for k in range(n):
		x_k = a + h * k
		K1 = f(x_k, Y[k])
		K2 = f(x_k + 1 / 3 * h, Y[k] + 1 / 3 * h * K1)
		K3 = f(x_k + 2 / 3 * h, Y[k] + 2 / 3 * h * K2)
		Y[k + 1] = Y[k] + h * (1 / 4 * K1 + 3 / 4 * K3)
	return Y

def solve(a, b, f, y0, start_n, eps, max_n):
	n = start_n
	Y_0 = evaluate_y(a, b, n, y0, f)
	while True:
		if n == max_n:
			return [pr[0] for pr in Y_0]
		n *= 2
		Y_1 = evaluate_y(a, b, n, y0, f)
		bad = False
		for i in range(1, len(Y_0)):
			bad |= abs(Y_0[i][0] - Y_1[2 * i][0]) > eps
		if not bad:
			return [pr[0] for pr in Y_1]
		Y_0 = Y_1


def deviations1(Y, Y_t, start_n, eps):
	for i in range(start_n + 1):
		ind = (len(Y) - 1) // start_n * i
		print('%d node: real = %.7f, evaluated = %.7f, diff = %.7f, diff/eps=%d' % 
			(i, Y_t[i], Y[ind], abs(Y_t[i] - Y[ind]), int(abs(Y_t[i] - Y[ind]) / eps * 100)), '%', sep = '')

def deviations2(Y, Y_t, start_n):
	for i in range(start_n + 1):
		ind = (len(Y) - 1) // start_n * i
		print('%d node: real = %.8f, evaluated = %.8f, diff = %.8f, diff/real=%d' % 
			(i, Y_t[i], Y[ind], abs(Y_t[i] - Y[ind]), int(abs(Y_t[i] - Y[ind]) / abs(Y_t[i]) * 100)), '%', sep = '')

def test1(a, b, y0, start_n, eps, y_t, f):
	Y = solve(a, b, f, y0, start_n, eps, 10 ** 9)
	print('eps = %.5f, len =  %d' % (eps, len(Y)))
	Y_t = [y_t(a + (b - a) / start_n * i) for i in range(start_n + 1)]
	deviations1(Y, Y_t, start_n, eps)

def test2(a, b, y0, start_n, max_n, y_t, f):
	Y = solve(a, b, f, y0, start_n, 0, max_n)
	print('len =  %d' % len(Y))
	Y_t = [y_t(a + (b - a) / start_n * i) for i in range(start_n + 1)]
	deviations2(Y, Y_t, start_n)

def rand(x, p):
	return x + random.randint(-p, p) / 100 * x

def run1(eps):
	test1(-1, 1, np.array([1.5]), 10, eps, y_t_1, f_1)
	
def run2(max_n):
	test2(1, 2, np.array([-1, -0.5]), 10, max_n, y_t_2, f_2)

def run2_1(p):
	test2(1, 2, np.array([rand(-1, p), -0.5]), 10, 10, y_t_2, f_2)

def run2_2(p):
	test2(1, 2, np.array([-1, rand(-0.5, p)]), 10, 10, y_t_2, f_2)

def process_command_line_args():
	run, param = None, None
	for arg in sys.argv[1:]:
		if arg.startswith('--run='):
			ar = arg[6:]
			if ar == '1':
				run = run1
			elif ar == '2':
				run = run2
			elif ar == '2.1':
				run = run2_1
			elif ar == '2.2':
				run = run2_2
			else:
				terminate('unexpected %s run was specified' % ar)
		elif arg.startswith('--param='):
			p = arg[8:]
			if p.find('.') == -1:
				param = int(p)
			else:
				param = float(p)
		else:
			terminate('unexpected %s arg was specified' % arg)
	if run is None:
		terminate('specify run')
	if param is None:
		terminate('specify param')
	return run, param

run, param = process_command_line_args()
run(param)




	


