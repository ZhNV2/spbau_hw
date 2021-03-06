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
	A = [[A[i][j] for j in range(len(A[i]))] for i in range(len(b))]
	b = [b[i] for i in range(len(b))]
	for i in range(n):
		x = [A[i][j] for j in range(i, n)]
		T = e(n - i)
		s = A[i][i]
		if abs(A[i][i]) < EPS:
			swaped = False;
			if full_mode:
				print("A[%d][%d] = 0, trying to swap rows" % (i, i))
			for j in range(i + 1, n):
				if abs(A[j][i]) > EPS:
					if full_mode:
						print("swapping rows %d and %d" % (i, j))
					A[i], A[j] = A[j], A[i]
					b[i], b[j] = b[j], b[i]
					swaped = True
					break
			if not swaped:
				terminate('det A = 0')
		for j in range(i + 1, n):
			if abs(A[j][i]) < EPS:
				continue
			tg = A[j][i] / (s + 0.)
			cos = (1 / (1 + tg ** 2)) ** 0.5
			sin = (1 - cos ** 2) ** 0.5
			if tg < 0:
				sin *= -1
			tmp = t(n - i, 0, j - i, cos, sin)
			if full_mode:
				print("setting A[%d][%d] to zero, A[%d][%d] = %.3f, tg = %.3f, sin = %.3f, cos = %.3f" 
					% (j, i, i, i, s, tg, sin, cos))
			s = s * cos + sin * A[j][i]
			T = mul(n - i, tmp, T)
		P = e(n)
		for i1 in range(n - i):
			for j1 in range(n - i):
				P[i + i1][i + j1] = T[i1][j1]
		
		A = mul(n, P, A)
		b = mul_v(n, P, b)

		if full_mode:
			print("--------- after setting %d-s column to zero, getting system -------" % i)
			print_system(A, b)
	
	x = [0 for i in range(n)]
	for i in range(n - 1, -1, -1):
		x[i] = b[i]
		for j in range(i + 1, n):
			x[i] -= A[i][j] * x[j]
		x[i] /= A[i][i]

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



if len(sys.argv) < 2:
	terminate('specify input file as first arg')
filename = sys.argv[1]
full_mode, noise_experiments = False, 0
for arg in sys.argv[2:]:
	if arg == '--full':
		full_mode = True
	elif arg.startswith('--noise'):
		noise_experiments = int(arg[7:])
	else:
		terminate('unexpected arg was specified')
n, A, b = read(filename)
x = solve(n, A, b, full_mode)
if noise_experiments == 0:
	print_solution(x)
else:
	noise(n, A, b, x, noise_experiments, full_mode)
