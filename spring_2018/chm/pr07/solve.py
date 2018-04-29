import matplotlib.pyplot as plt
import sys
import math
import random


#------------- from previous tasks --------------#

EPS = 1e-12

def terminate(msg):
	print('program finished with error:')
	print(msg)
	exit()

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
	return x + random.randint(-4, 4) * x / 100.
	
def noise(n, A, b, x, noise_experiments, full_mode):
	A_changes = []
	x_changes = []
	sensitivity = []
	for tmp in range(noise_experiments):
		CA = [[make_noise(A[i][j]) for j in range(n)] for i in range(n)]
		deltaA = [[abs(A[i][j] - CA[i][j]) / A[i][j] for j in range(n)] for i in range(n)]
		if full_mode:
			print('-------noise matrix')
			print_system(deltaA, b)
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

	

#------------- functions for current task --------------#
def standard(n, x, a_, b_):
	return x ** n

def legendre(n, x, a_, b_):
	x = 1.0 * (2 * x - (a_ + b_)) / (b_ - a_)
	v = [1] * max(3, n + 1)
	v[1] = x
	for i in range(1, n):
		v[i + 1] = 1.0 * ((2 * i + 1) * x * v[i] - i * v[i - 1]) / (i + 1)
	return v[n]


def process_command_line_args():
	filename, full_mode, plot, deg, phi = None, False, False, None, None
	for arg in sys.argv[1:]:
		if arg == '-f' or arg == '--full':
			full_mode = True
		elif arg == '-p' or arg == '--plot':
			plot = True
		elif arg.startswith('--deg='):
			deg = int(arg[6:])
		elif arg.startswith('--input='):
			filename = arg[8:]
		elif arg.startswith('--phi='):
			if arg[6:] == 'usual':
				phi = standard
			else:
				phi = legendre
		else:
			terminate('unexpected %s arg was specified' % arg)
	if filename is None:
		terminate('specify file with input')
	if deg is None:
		terminate('specify deg')
	return filename, full_mode, plot, deg, phi


def read(filename):
	tokens = []
	with open(filename) as input_data_file:
		tokens = input_data_file.read().split()
	pos = 0
	n = int(tokens[pos])
	pos += 1
	X, Y = [], []
	for _ in range(n):
		X.append(float(tokens[pos]))
		pos += 1
	for _ in range(n):
		Y.append(float(tokens[pos]))
		pos += 1
	return n, X, Y

def build_matrix(X, n, a_, b_, phi):
	A = [[0 for _ in range(n)] for _ in range(n)]
	for i in range(n):
		for j in range(n):
			A[i][j] = phi(j, X[i], a_, b_)
	return A

def deviations(X, Y, P):
	min_dev = 10 ** 20
	max_dev = -10 ** 20
	sum_dev = 0
	for i in range(n):
		dev = abs(Y[i] - P[i])
		min_dev = min(min_dev, dev)
		max_dev = max(max_dev, dev)
		sum_dev += dev
	return min_dev, max_dev, 1.0 * sum_dev / len(X)


#------------- main --------------#

filename, full_mode, plot, m, phi = process_command_line_args()
n, X, Y = read(filename)
a_, b_ = min(X), max(X)



if full_mode:
	print("---- input data ----")
	print('n = %d, m = %d' % (n, m))
	print('X =', X)
	print('Y =', Y)



indexes = [(n - 1) // m * i for i in range(m + 1)]
bX = [X[i] for i in indexes]
bY = [Y[i] for i in indexes]

A = build_matrix(bX, m + 1, a_, b_, phi)
a = solve(m + 1, A, bY, full_mode)

print('--------- result polynomial:----------')
print('%fphi_{0}' % a[0], end = '')
for i in range(1, m + 1):
	print('+ %fphi_{%d}' % (a[i], i), end = '')
print()


P = [sum([a[j] * phi(j, x, a_, b_) for j in range(m + 1)]) for x in X]

noise(m + 1, A, bY, a, 10, full_mode)

print('--------- deviations ----------')
min_dev, max_dev, avg_dev = deviations(X, Y, P)
print('min = %.15f' % min_dev)
print('max = %.15f' % max_dev)
print('avg = %.15f' % avg_dev)

if plot:
	plt.plot(X, Y, color = 'blue') 
	plt.plot(X, P, color = 'orange')
	plt.show()



