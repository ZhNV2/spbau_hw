import matplotlib.pyplot as plt
import math
import numpy as np

def f(x, pr):
	y, t = pr[0], pr[1]
	return [t, (-(2 - x) * t - 2 * y - x ** 0.5) / (2 * x * (x + 2))]

def y_t(x):
	return x ** 0.5


def evaluate_y(a, b, n, y0):
	P = [[] for _ in range(n + 1)]
	P[0] = y0
	h = (b - a) / n
	for k in range(n):
		x_k = a + h * k
		K1 = f(x_k, Y[k])
		K2 = f(x_k + 1 / 3 * h, Y[k] + 1 / 3 * h * K1)
		K3 = f(x_k + 2 / 3 * h, Y[k] + 2 / 3 * h * K2)
		Y[k + 1] = Y[k] + h * (1 / 4 * K1 + 3 / 4 * K3)
	return Y

def solve(a, b, f, y0, start_n, eps):
	n = start_n
	Y_0 = evaluate_y(a, b, n, y0)
	while True:
		n *= 2
		Y_1 = evaluate_y(a, b, n, y0)
		bad = False
		for i in range(1, len(Y_0)):
			bad |= abs(Y_0[i] - Y_1[2 * i]) > eps
		if not bad:
			return Y_1
		Y_0 = Y_1


def deviations(Y, Y_t, start_n, eps):
	for i in range(start_n + 1):
		ind = (len(Y) - 1) // start_n * i
		print('%d node: real = %.7f, evaluated = %.7f, diff = %.7f, diff/eps=%d' % 
			(i, Y_t[i], Y[ind], abs(Y_t[i] - Y[ind]), int(abs(Y_t[i] - Y[ind]) / eps * 100)), '%', sep = '')

def test(a, b, y0, start_n, eps):
	Y = solve(a, b, f, y0, start_n, eps)
	print('eps = %.5f, len =  %d' % (eps, len(Y)))
	Y_t = [y_t(a + (b - a) / start_n * i) for i in range(start_n + 1)]
	deviations(Y, Y_t, start_n, eps)

test(-1, 1, 1.5, 10, 0.001)
test(-1, 1, 1.5, 10, 0.0001)
test(-1, 1, 1.5, 10, 0.00001)


