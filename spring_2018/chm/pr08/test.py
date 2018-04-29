import sys
import random

EPS = 1e-12

def terminate(msg):
	print('program finished with error:')
	print(msg)
	exit()

def read(filename):
	lines = []
	with open(filename) as input_data_file:
		lines = input_data_file.readlines()
	n = len(lines)
	A = []
	b = []
	for line in lines:
		lst = list(map(float, line.split()))
		A.append(lst[:len(lst) - 1])
		b.append(lst[-1])
	return n, A, b

def e(n):
	return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

def t(n, i, j, cos, sin):
	M = e(n)
	M[i][i] = cos
	M[i][j] = sin
	M[j][i] = -sin
	M[j][j] = cos
	return M

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

def print_system(A, b):
	for i in range(len(A)):
		row = A[i]
		for j in range(len(row) - 1):
			print("%.6fx[%d] + " % (row[j], j), end = '')
		print("%.6fx[%d]" % (row[-1], len(row) - 1), end = '')
		print(" = %.6f" % b[i])
	print()

def print_solution(x):
	print("--------- solution x: ----------")
	for i in range(len(x)):
		print("x[%d] = %.3f, " % (i, x[i]), end = '')
	print('')



def solve(n, A, b, full_mode):
	if full_mode:
		print('---------- algorithm starts from system ----------')
		print_system(A, b)
	l = e(n)
	A = [[A[i][j] for j in range(n)] for i in range(n)]
	d = [[0 for _ in range(n)] for _ in range(n)]
	u = e(n)
	for i in range(n):
		to_swap = -1
		for j in range(i, n):

			for t in range(0, i):
				l[i][t] = A[j][t]
				for k in range(t):
					l[i][t] -= l[i][k] * d[k][k] * u[k][t]
				l[i][t] /= d[t][t]
			d[i][i] = A[j][i]
			for k in range(i):
				d[i][i] -= l[i][k] * d[k][k] * u[k][i]

			if abs(d[i][i]) > EPS:
				to_swap = j
				break
		if to_swap == -1:
			terminate('det A = 0')
		elif i != to_swap:
			A[i], A[to_swap] = A[to_swap], A[i]
			b[i], b[to_swap] = b[to_swap], b[i]
			if full_mode:
				print('swapping rows %d and %d' % (i, to_swap))
		
		for j in range(i + 1, n):
			sum = 0
			for k in range(0, i):
				sum += l[i][k] * d[k][k] * u[k][j]
			u[i][j] = (A[i][j] - sum) / d[i][i]
		
	
	x1 = [0] * n
	x = [0] * n
	for i in range(n):
		x1[i] = b[i]
		for j in range(i):
			x1[i] -= l[i][j] * x1[j]
	for i in range(n):
		x1[i] /= d[i][i]
	for i in range(n - 1, -1, -1):
		x[i] = x1[i]
		for j in range(i + 1, n):
			x[i] -= u[i][j] * x[j]

	if full_mode:
		print_solution(x)
	return x

def norm_v(n, x):
	return sum([abs(t) for t in x])

def norm_A(n, A):
	return max([sum([abs(A[i][j]) for i in range(n)]) for j in range(n)])

def make_noise(x):
	return x + random.randint(-2, 2) * x / 100.
	
def noise(n, A, b, x, noise_experiments, full_mode):
	A_changes = []
	x_changes = []
	sensitivity = []
	for tmp in range(noise_experiments):
		CA = [[make_noise(A[i][j]) for j in range(n)] for i in range(n)]
		Cb = [b[i] for i in range(len(b))]
		if full_mode:
			print("--- experiment #%d, system is" % (tmp + 1))
			print_system(CA, Cb)
		Cx = solve(n, CA, Cb, full_mode)
		dA = [[abs(A[i][j] - CA[i][j]) for j in range(n)] for i in range(n)]
		dx = [abs(x[i] - Cx[i]) for i in range(n)]
		A_changes.append(1.0 * norm_A(n, dA) / norm_A(n, A))
		x_changes.append(1.0 * norm_v(n, dx) / norm_v(n, x))
		sensitivity.append(1.0 * x_changes[-1] / A_changes[-1])
		if full_mode:
			print("A relative change = %.3f" % A_changes[-1])
			print("x relative change = %.3f" % x_changes[-1])

	max_x_change = max(x_changes) * 100
	avg_x_change = 100.0 * sum(x_changes) / noise_experiments
	min_x_change = min(x_changes) * 100
	
	max_A_change = max(A_changes) * 100
	avg_A_change = 100.0 * sum(A_changes) / noise_experiments
	min_A_change = min(A_changes) * 100 

	max_sensitivity = max(sensitivity)
	avg_sensitivity = 1.0 * sum(sensitivity) / noise_experiments
	min_sensitivity = min(sensitivity) 
	print("--------- noise experiment results ---------")
	print("max A relative change = %.5f, max x relative change = %.5f, max sensitivity = %.5f" % (max_A_change, max_x_change, max_sensitivity))
	print("avg A relative change = %.5f, avg x relative change = %.5f, avg sensitivity = %.5f" % (avg_A_change, avg_x_change, avg_sensitivity))
	print("min A relative change = %.5f, min x relative change = %.5f, min sensitivity = %.5f" % (min_A_change, min_x_change, min_sensitivity))


def sqrt(x):
	return x ** 0.5

x = [-1, -1 / sqrt(5), 1 / sqrt(5), 1]
b = [2, 0, 2 / 3, 0]
A = [[0 for _ in range(4)] for _ in range(4)]
for i in range(4):
	for j in range(4):
		A[i][j] = x[j] ** i

As = solve(4, A, b, False)
print(*As)
