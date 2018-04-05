import random
import sys


EPS = 1e-12

def terminate(msg):
	print('program finished with error:')
	print(msg)
	exit()


def read(filename):
	lines = []
	with open(filename) as input_data_file:
		lines = input_data_file.readlines()
	
	M = []
	for line in lines:
		M.append(list(map(float, line.split())))
	x0 = M.pop()
	xs = M.pop()
	n = []
	A = []
	pos = 0
	while pos < len(M):
		cur_n = 0
		while pos + cur_n < len(M) and abs(M[pos][pos + cur_n]) > EPS:
			cur_n += 1
		cur_A = [[M[pos + i][pos + j] for j in range(cur_n)] for i in range(cur_n)]
		n.append(cur_n)
		A.append(cur_A)
		pos += cur_n
	x_0 = []
	x_s = []
	sum = 0
	for i in range(len(n)):
		x_0.append(x0[sum : sum + n[i]])
		x_s.append(xs[sum : sum + n[i]])
		sum += n[i]
	return n, A, x_s, x_0

def concat(lst):
	return sum(lst, [])

def norm(n, x_1, x_2):
	N, x1, x2 = sum(n), concat(x_1), concat(x_2)
	return sum([(x1[i] - x2[i]) ** 2 for i in range(N)]) ** 0.5

def mul_v(n, a, x):
	y = [0 for i in range(n)]
	for i in range(n):
		for j in range(n):
			y[i] += a[i][j] * x[j]
	return y


def seidel(n, A, x_s, x_0, eps, full_mode):
	N = len(n)
	b = [mul_v(n[i], A[i], x_s[i]) for i in range(N)]
	k = 0
	while True:
		x_1 = [0] * N
		if full_mode:
			print('------ x[%d] = -----' % k)
			print(concat(x_0))
		for num in range(N):
			x_1[num] = [0] * n[num]
			for i in range(n[num]):
				for j in range(i):
					x_1[num][i] -= 1.0 * A[num][i][j] / A[num][i][i] * x_1[num][j]
				for j in range(i + 1, n[num]):
					x_1[num][i] -= 1.0 * A[num][i][j] / A[num][i][i] * x_0[num][j]
				x_1[num][i] += 1.0 * b[num][i] / A[num][i][i]
		if norm(n, x_0, x_1) < eps:
			return x_1, k
		x_0 = x_1
		k = k + 1

def scalar(n, x_1, x_2):
	return sum([x_1[i] * x_2[i] for i in range(n)])

def g(x, A, b, n):
	t = mul_v(n, A, x)
	return [t[i] - b[i] for i in range(n)]

def grad(n, A, x_s, x_0, eps, full_mode):
	N = len(n)
	b = [mul_v(n[i], A[i], x_s[i]) for i in range(N)]
	k = 0
	g_0 = [g(x_0[i], A[i], b[i], n[i]) for i in range(N)]
	while True:
		alp = []
		if full_mode:
			print('------ x[%d] = -----' % k)
			print(concat(x_0))
			print('------ gradient[%d] = -----' %k)
			print(concat(g_0))
		for i in range(N):
			if scalar(n[i], g_0[i], g_0[i]) < EPS:
				alp.append(1)
			else:
				alp.append(scalar(n[i], g_0[i], g_0[i]) / 2. / scalar(n[i], mul_v(n[i], A[i], g_0[i]), g_0[i]))
		x_1 = [[x_0[i][j] - alp[i] * g_0[i][j] for j in range(n[i])] for i in range(N)]
		if norm(n, x_0, x_1) < eps:
			return x_1, k
		x_0 = x_1
		g_0 = [g(x_0[i], A[i], b[i], n[i]) for i in range(N)]
		k = k + 1


if len(sys.argv) < 2:
	terminate('specify input file as first arg')
filename = sys.argv[1]
full_mode, method, eps = False, 'seidel', 0.001
for arg in sys.argv[2:]:
	if arg == '--full':
		full_mode = True
	elif arg.startswith('--method='):
		method = arg[9:]
	elif arg.startswith('--eps='):
		eps = float(arg[6:])
	else:
		terminate('unexpected arg was specified')

n, A, x_s, x_0 = read(filename)

if full_mode:
	print('---- initial n -----')
	print(n)
	print('---- initial A -----')
	print(A)
	print('---- initial x_s -----')
	print(x_s)
	print('---- initial x_0 -----')
	print(x_0)

if method == 'seidel':
	x, k = seidel(n, A, x_s, x_0, eps, full_mode)
elif method == 'grad':
	x, k = grad(n, A, x_s, x_0, eps, full_mode)
else:
	terminate('unknown method, use seidel or grad')

print('---- given solution x* ------')
print(concat(x_s))
print('---- given start x0 ------')
print(concat(x_0))
print('---- found solution in %d iterations ------' % k)
print(concat(x))


