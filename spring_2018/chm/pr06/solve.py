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

	
def deviations(X, Y, P):
	min_dev = 10 ** 20
	max_dev = -10 ** 20
	sum_dev = 0
	for i in range(len(X)):
		dev = abs(Y[i] - P[i])
		min_dev = min(min_dev, dev)
		max_dev = max(max_dev, dev)
		sum_dev += dev
	return min_dev, max_dev, 1.0 * sum_dev / len(X)

#------------- functions for current task --------------#

def f(x):
	return 1.0 / (1.0 + 25 * x * x)

#def f(x):
# 	return x ** 0.5 - math.cos(5 * x)

def get(splines, x, X):
	for i in range(len(X) - 1):
		if X[i] <= x and x <= X[i + 1]:
			return splines[i][0] + splines[i][1] * (x - X[i]) + splines[i][2] * (x - X[i]) ** 2

def get3(splines, x, X):
	for i in range(len(X) - 1):
		if X[i] <= x and x <= X[i + 1]:
			return splines[4 * i] + splines[4 * i + 1] * (x - X[i]) + splines[4 * i + 2] * (x - X[i]) ** 2 + splines[4 * i + 3] * (x - X[i]) ** 3

n = 10
a, b = -1, 1

X = [a + i * (b - a + 0.) / n for i in range(n + 1)]
#X = [(a + b) / 2.  + (b - a) / 2. * (math.cos((2 * k - 1) * math.pi / 2. / (n + 1))) for k in range(1, n + 2)]

Y = [f(x) for x in X]

testN = 50
testX = [a + i * (b - a + 0.) / testN for i in range(testN + 1)]
testY = [f(x) for x in testX]

splines = []
A = [[0] * 3] * 3
toT = 4
for i in range(3):
	A[i] = [1, X[toT + i] - X[toT], (X[toT + i] - X[toT]) ** 2]

B = Y[toT: toT + 3]
x = solve(3, A, B, False)
splines.append(x)
for k in range(toT - 1, -1, -1):
	for j in range(2):
		A[j] = [1, X[k + j] - X[k], (X[k + j] - X[k]) ** 2]
	A[2] = [0, 1, 2 * (X[k + 1] - X[k])]
	x = splines[0]
	B = [Y[k], Y[k + 1], x[1] + 2 * x[2] * (X[k + 1] - X[k + 1])]
	x = solve(3, A, B, False)
	splines = [x] + splines
for k in range(toT + 2, n):
	for j in range(2):
		A[j] = [1, X[k + j] - X[k], (X[k + j] - X[k]) ** 2]
	A[2] = [0, 1, 0]
	x = splines[-1]
	B = [Y[k], Y[k + 1], x[1] + 2 * x[2] * (X[k] - X[toT if k == toT + 2 else k - 1])]
	x = solve(3, A, B, False)
	splines.append(x)


A = [[0 for _ in range(4 * n)] for _ in range(4 * n)]
B = [0] * (4 * n)
pos = 0
for k in range(n):
	for i in range(2):
		for st in range(4):
			A[pos][4 * k + st] = (X[k + i] - X[k]) ** st
		B[pos] = Y[k + i]
		pos += 1
for k in range(1, n):
	A[pos][4 * (k - 1) + 1] = 1
	A[pos][4 * (k - 1) + 2] = 2 * (X[k] - X[k - 1])
	A[pos][4 * (k - 1) + 3] = 3 * (X[k] - X[k - 1]) ** 2
	A[pos][4 * k + 1] = -1
	A[pos][4 * k + 2] = -2 * (X[k] - X[k])
	A[pos][4 * k + 3] = -3 * (X[k] - X[k]) ** 2
	pos += 1
for k in range(1, n):
	A[pos][4 * (k - 1) + 2] = 2
	A[pos][4 * (k - 1) + 3] = 6 * (X[k] - X[k - 1])
	A[pos][4 * k + 2] = -2
	A[pos][4 * k + 3] = -6 * (X[k] - X[k])
	pos += 1
A[pos][4 * 0 + 2] = 2
A[pos][4 * 0 + 3] = 6 * (X[0] - X[0])
pos += 1
A[pos][4 * (k - 1) + 2] = 2
A[pos][4 * (k - 1) + 3] = 6 * (X[n] - X[n - 1])
pos += 1

splines3 = solve(4 * n, A, B, False)


testP = [get(splines, x, X[:toT + 1] + X[toT + 2:]) for x in testX]
testG = [get3(splines3, x, X) for x in testX]


print('--------- deviations ----------')
min_dev, max_dev, avg_dev = deviations(testX, testY, testP)
print('min = %.15f' % min_dev)
print('max = %.15f' % max_dev)
print('avg = %.15f' % avg_dev)

plt.plot(testX, testY, color = 'blue') 
plt.plot(testX, testP, color = 'orange')
#plt.plot(testX, testG, color = 'green')
plt.show()



