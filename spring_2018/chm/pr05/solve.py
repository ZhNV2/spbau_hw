import matplotlib.pyplot as plt
import sys
import math
import random


#------------- from previous tasks --------------#

def terminate(msg):
	print('program finished with error:')
	print(msg)
	exit()

#------------- functions for current task --------------#


def deviations(X, Y, P):
	max_abs_dev = -10 ** 20
	max_rel_dev = -10 ** 20
	for i in range(len(X)):
		dev = abs(Y[i] - P[i])
		max_abs_dev = max(max_abs_dev, dev)
		max_rel_dev = max(max_rel_dev, dev / Y[i])
	return max_abs_dev, max_rel_dev

def deviations2(X, Y, P):
	max_abs_dev = -10 ** 20
	sub_abs_dev = 0
	for i in range(len(X)):
		dev = abs(Y[i] - P[i])
		max_abs_dev = max(max_abs_dev, dev)
		sub_abs_dev += dev
	return max_abs_dev, sub_abs_dev / len(X)

def process_command_line_args():
	filename, full_mode, m, grid, plot = None, False, None, None, False
	for arg in sys.argv[1:]:
		if arg == '-f' or arg == '--full':
			full_mode = True
		elif arg == '-p' or arg == '--plot':
			plot = True
		elif arg.startswith('--deg='):
			m = int(arg[6:])

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
	if grid is None:
		terminate('specify grid type')
	return filename, full_mode, m, grid, plot

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
	return [closest(x, uni) for x in che]

def subseq(a, indexes):
	return [a[i] for i in indexes]

def calc_newton(xes, a, X):
	yes = []
	for x in xes:
		res = 0
		for j in range(len(a)):
			pp = 1
			for i in range(j):
				pp *= (x - X[i])
			res += a[j] * pp
		yes.append(res)
	return yes


def newton(n, X, Y):
	a = [Y[0]]
	for k in range(1, n + 1):
		wk = 1
		for i in range(k):
			wk *= (X[k] - X[i])
		pk = calc_newton([X[k]], a, X)[0]
		a.append((Y[k] - pk) / wk)
	return a

#------------- main --------------#

filename, full_mode, m, grid, plot = process_command_line_args()
n, X, Y = read(filename)
a_, b_ = min(X), max(X)
if full_mode:
	print('n =', n, ',a_ =', a_, ',b_ =', b_)

indexes = grid(n, a_, b_, m)
if full_mode:
	print('chosen xes:', indexes)


pX = subseq(X, indexes)
pY = subseq(Y, indexes)

a = newton(m, pX, pY)
if full_mode:
	print('newtons coefs:', a)
P = calc_newton(X, a, pX)

print('--------- deviations ----------')
max_abs_dev, avg_abs_dev = deviations2(X, Y, P)
print('max absolute deviation = %.15f' % max_abs_dev)
print('avg absolute deviation = %.15f' % avg_abs_dev)

if plot:
	plt.plot(X, Y, color = 'blue') 
	plt.plot(X, P, color = 'orange')
	plt.show()



