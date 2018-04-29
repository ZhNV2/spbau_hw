import matplotlib.pyplot as plt
import math

def f(x, y):
	return y / (x + 2) + x ** 2 + 2 * x

def y_t(x):
	return (1 + 0.5 * x ** 2) * (x + 2)

def evaluate_y(a, b, n, y0):
	Y = [0] * (n + 1)
	Y[0] = y0
	for k in range(n):
		x_k = a + (b - a) / n * k
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
		ind = (len(Y) - 1) / start_n * i
		print('%d node: real = %.5f, evaluated = %.5f, diff = %.5f, eps = %.5f' % (i, Y_t[ind], Y[ind], abs(Y_t[ind] - Y[ind]), eps))

def test(a, b, y0, start_n, eps):
	Y = solve(a, b, f, y0, start_n, eps)
	Y_t = [y_t(a + (b - a) / start_n * i) for i in range(start_n + 1)]
	deviations(Y, Y_t, start_n, eps)


test(-1, 1, 1.5, 10, 0.01)


