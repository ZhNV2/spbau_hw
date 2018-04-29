import matplotlib.pyplot as plt
import sys
import math


#------------- from previous tasks --------------#

EPS = 1e-6

def terminate(msg):
	print('program finished with error:')
	print(msg)
	exit()

def closest(x, nums):
	mi = 10 ** 20
	ind = -1
	for i in range(len(nums)):
		if abs(mi - x) > abs(nums[i] - x):
			mi = nums[i]
			ind = i
	return ind

def uniform(n, a_, b_, m):
	return [int(n * i / m) for i in range(m + 1)]

def сhebyshevX(n, a_, b_, m):
	uni = [a_ + i * (b_ - a_ + 0.) / n for i in range(n + 1)]
	che = [(a_ + b_) / 2.  + (b_ - a_) / 2. * (math.cos((2 * k - 1) * math.pi / 2. / (m + 1))) for k in range(1, m + 2)]
	nodes = [closest(x, uni) for x in che]
	nodes[0] = 0
	nodes[-1] = n
	return nodes


def subseq(a, indexes):
	return [a[i] for i in indexes]

def mul_v(a, x):
	n = len(a)
	if n == 0:
		return []
	m = len(a[0])
	y = [0 for i in range(n)]
	for i in range(n):
		for j in range(m):
			y[i] += a[i][j] * x[j]
	return y

def deviations(X, Y, P):
	max_abs_dev = -10 ** 20
	sub_abs_dev = 0
	for i in range(len(X)):
		dev = abs(Y[i] - P[i])
		max_abs_dev = max(max_abs_dev, dev)
		sub_abs_dev += dev
	return max_abs_dev, sub_abs_dev / len(X)

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

def e(n):
	return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


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
	return n - 1, X, Y


#------------- functions for current task --------------#

def tridiagonal_matrix_solution(A, f):
	n = len(f)
	a = [0] + [A[i][i - 1] for i in range(1, n)]
	b = [A[i][i] for i in range(n)]
	c = [A[i][i + 1] for i in range(0, n - 1)] + [0]
	alpha = [0] * n
	betta = [0] * n
	alpha[0] = -c[0] / b[0]
	betta[0] = f[0] / b[0]
	for i in range(1, n):
		alpha[i] = -c[i] / (b[i] + a[i] * alpha[i - 1])
		betta[i] = (f[i] - a[i] * betta[i - 1]) / (b[i] + a[i] * alpha[i - 1])
	x = [0] * n
	x[n - 1] = betta[n - 1]
	for i in range(n - 2, -1, -1):
		x[i] = alpha[i] * x[i + 1] + betta[i]
	return x

class Spline():

	def der(self, k, x):
		return 3 * self.a[k][0] * x * x + 2 * self.a[k][1] * x + self.a[k][2]

	def der2(self, k, x):
		return 6 * self.a[k][0] * x + 2 * self.a[k][1]

	def test(self, full_mode):
		for k in range(1, len(self.X) - 1):
			der = abs(self.der(k, self.X[k]) - self.der(k + 1, self.X[k]))
			if der < EPS:
				if full_mode:
					print('first derivatives on x[%d] are not equal from opposite sides, diff=%.10f' % (k, der))
				return False
			der2 = abs(self.der2(k, self.X[k]) - self.der2(k + 1, self.X[k]))
			if der2 < EPS:
				if full_mode:
					print('second derivatives on x[%d] are not equal from opposite sides, diff=%.10f' % (k, der2))
				return False
		return True

	def build_b(self, n, h, Y):
		H = [[0 for _ in range(n + 1)] for _ in range(n - 1)]
		for k in range(1, n):
			H[k - 1][k - 1] = 1 / h[k]
			H[k - 1][k] = -(1 / h[k] + 1 / h[k + 1])
			H[k - 1][k + 1] = 1 / h[k + 1]

		return mul_v(H, Y)

	def build_coef_by_m(self, n, m, Y, h):
		self.a = [[]]
		for k in range(1, n + 1):
			a = [0] * 4
			a[0] = (m[k] - m[k - 1]) / 6 / h[k]
			a[1] = m[k - 1] / 2
			a[2] = (Y[k] - Y[k - 1]) / h[k] - h[k] / 6 * (m[k] + 2 * m[k - 1])
			a[3] = Y[k - 1]
			self.a.append(a)

	def evaluate(self, x):
		for k in range(1, len(self.X)):
			if self.X[k - 1] <= x and x <= self.X[k]:
				return sum([self.a[k][i] * (x - self.X[k - 1]) ** (3 - i) for i in range(4)])

class CustomSpline(Spline):
	def __init__(self, X, Y, ind, v, full_mode):
		if ind == 0 or ind == len(X) - 1:
			terminate('can\'t exclude outermost x')
		cpX, cpY = [x for x in X], [y for y in Y]
		self.X = cpX
		cpX.pop(ind)
		cpY.pop(ind)
		n = len(X) - 2
		h = [0] + [cpX[i] - cpX[i - 1] for i in range(1, n + 1)]
		b = self.build_b(n, h, cpY) + [v]
		A = [[0 for _ in range(n + 1)] for _ in range(n + 1)]
		for k in range(1, n):
			A[k - 1][k - 1] = h[k] / 6
			A[k - 1][k] = (h[k] + h[k + 1]) / 3
			A[k - 1][k + 1] = h[k + 1] / 6
		A[n - 1][n] = 1
		
		d = X[ind] - cpX[ind - 1]

		A[n][ind - 1] = -(d ** 3) / 6 / h[ind] + d ** 2 / 2 - h[ind] / 3 * d
		A[n][ind] = (d ** 3) / 6 / h[ind] - h[ind] / 6 * d
		b = b + [Y[ind] - (cpY[ind] - cpY[ind - 1]) / h[ind] * d - cpY[ind - 1]]

		m = solve(n + 1, A, b, full_mode)
		self.build_coef_by_m(n, m, cpY, h)


class NaturalCubicSpline(Spline):

	def __init__(self, X, Y):
		self.X = [x for x in X]
		
		n = len(X) - 1
		h = [0] + [X[i] - X[i - 1] for i in range(1, n + 1)]
		
		A = [[0 for _ in range(n - 1)] for _ in range(n - 1)]
		for k in range(1, n):
			if k != 1:
				A[k - 1][k - 2] = h[k] / 6
			A[k - 1][k - 1] = (h[k] + h[k + 1]) / 3
			if k != n - 1:
				A[k - 1][k] = h[k + 1] / 6

		b = self.build_b(n, h, Y)
		m = [0] + tridiagonal_matrix_solution(A, b) + [0]

		self.build_coef_by_m(n, m, Y, h)


				
def process_command_line_args():
	filename, full_mode, m, grid, ex, y2b, plot, type_ = None, False, None, None, None, None, False, None
	for arg in sys.argv[1:]:
		if arg == '-f' or arg == '--full':
			full_mode = True
		elif arg == '-p' or arg == '--plot':
			plot = True
		elif arg.startswith('--type='):
			type_ = arg[7:]
		elif arg.startswith('--deg='):
			m = int(arg[6:])
		elif arg.startswith('--ex='):
			ex = int(arg[5:])
		elif arg.startswith('--y2b='):
			y2b = float(arg[6:])
		elif arg.startswith('--grid='):
			pol = arg[7:]
			if pol == 'uniform':
				grid = uniform
			elif pol == 'chebyshev':
				grid = сhebyshevX
			else:
				terminate('unexpected grid')
		elif arg.startswith('--input='):
			filename = arg[8:]
		else:
			terminate('unexpected %s arg was specified' % arg)
	if filename is None:
		terminate('specify file with input')
	if m is None:
		terminate('specify polinomial degree')
	if type_ == 'natural':
		if ex is not None:
			terminate('ex parameter only for custom not for natural spline')
		if y2b is not None:
			terminate('y2b parameter only for custom not for natural spline')
	elif type_ == 'custom':
		if ex is None:
			terminate('specify node for exclusion (ex) for custom spline')
		if y2b is None:
			terminate('specify second derivative in right edge(y2b) for custom spline')
	else:
		terminate('unexpected %s spline type' % type_)
	return filename, full_mode, m, grid, ex, y2b, plot, type_

filename, full_mode, m, grid, ex, y2b, plot, type_ = process_command_line_args()
n, X, Y = read(filename)
a_, b_ = min(X), max(X)
if full_mode:
	print('n =', n, ',a_ =', a_, ',b_ =', b_, ',m = ', m)

indexes = sorted(grid(n, a_, b_, m))
if full_mode:
	print('chosen xes:', indexes)

pX = subseq(X, indexes)
pY = subseq(Y, indexes)

if type_ == 'natural':
	spline = NaturalCubicSpline(pX, pY)
elif type_ == 'custom':
	spline = CustomSpline(pX, pY, ex, y2b, full_mode)

if not spline.test(full_mode):
	terminate('built spline is incorrect')

P = [spline.evaluate(x) for x in X]

print('--------- deviations ----------')
max_abs_dev, avg_abs_dev = deviations(X, Y, P)
print('max absolute deviation = %.15f' % max_abs_dev)
print('avg absolute deviation = %.15f' % avg_abs_dev)

if plot:
	plt.plot(X, Y, color = 'blue') 
	plt.plot(X, P, color = 'orange')
	plt.show()
