import random

INPUT_FILE_NAME = 'input'

def read():
	lines = []
	with open(INPUT_FILE_NAME) as input_data_file:
		lines = input_data_file.readlines()
	n = len(lines)
	A = []
	for line in lines:
		A.append(list(map(float, line.split())))
	return n, A

def e(n):
	return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

def norm(n, x_1, x_2):
	return sum([(x_1[i] - x_2[i]) ** 2 for i in range(n)]) ** 0.5

def mul(n, a, b):
	c = [[0 for j in range(n)] for i in range(n)]
	for i in range(n):
		for j in range(n):
			for k in range(n):
				c[i][j] += a[i][k] * b[k][j]
	return c

def mul_v(n, a, x):
	y = [0 for i in range(n)]
	for i in range(n):
		for j in range(n):
			y[i] += a[i][j] * x[j]
	return y


def seidel(n, A, x_s, x_0, eps):
	E = e(n)
	A_l = [[A[i][j] if i > j else 0 for j in range(n)] for i in range(n)]
	D = [[1. / A[i][j] if i == j else 0 for j in range(n)] for i in range(n)]
	A_u = [[A[i][j] if i < j else 0 for j in range(n)] for i in range(n)]

	K = mul(n, D, A_l)
	K = [[-K[i][j] for j in range(n)] for i in range(n)]
	L = mul(n, D, A_u)
	L = [[-L[i][j] for j in range(n)] for i in range(n)]
	b = mul_v(n, A, x_s)
	g = mul_v(n, D, b)

	k = 0
	while True:
		x_1 = [0] * n
		for i in range(n):
			for j in range(n):
				x_1[i] += K[i][j] * x_1[j]
			for j in range(n):
				x_1[i] += L[i][j] * x_0[j]
			x_1[i] += g[i]
		if norm(n, x_0, x_1) < eps:
			return x_1, k
		x_0 = x_1
		k = k + 1

def scalar(n, x_1, x_2):
	return sum([x_1[i] * x_2[i] for i in range(n)])

def g(x, A, b, n):
	t = mul_v(n, A, x)
	return [t[i] - b[i] for i in range(n)]

def grad(n, A, x_s, x_0, eps):
	b = mul_v(n, A, x_s)
	k = 0
	g_0 = g(x_0, A, b, n)
	while True:
		alp = scalar(n, g_0, g_0) / 2. / scalar(n, mul_v(n, A, g_0), g_0)
		x_1 = [x_0[i] - alp * g_0[i] for i in range(n)]
		if norm(n, x_0, x_1) < eps:
			return x_1, k
		x_0 = x_1
		g_0 = g(x_0, A, b, n)
		k = k + 1


def main():
	n, A = read()
	x_s = [1, -2, 3, -8, -6]
	x_0 = [-8, -6, -4, 4, -5]
	eps = 0.1
	x_seidel, k_seidel = seidel(n, A, x_s, x_0, eps)
	x_grad, k_grad =grad(n, A, x_s, x_0, eps)
	print('seidel')
	print(x_seidel)
	print(k_seidel)
	print('grad')
	print(x_grad)
	print(k_grad)


main()


