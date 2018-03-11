import sys

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

def solve(n, A, b):
	CA = [[A[i][j] for j in range(n)] for i in range(n)]
	for i in range(n):
		x = [A[i][j] for j in range(i, n)]
		T = e(n - i)
		s = A[i][i]
		for j in range(i + 1, n):
			tg = A[j][i] / (s + 0.)
			cos = (1 / (1 + tg ** 2)) ** 0.5
			sin = (1 - cos ** 2) ** 0.5
			if tg < 0:
				sin *= -1
			tmp = t(n - i, 0, j - i, cos, sin)
			s = s * cos + sin * A[j][i]
			T = mul(n - i, tmp, T)
		P = e(n)
		for i1 in range(n - i):
			for j1 in range(n - i):
				P[i + i1][i + j1] = T[i1][j1]
		
		A = mul(n, P, A)
		b = mul_v(n, P, b)


	for i in range(n):
		for j in range(n):
			print('{0:6f} '.format(A[i][j]), end='')
		print()
	print(b)

	print("-------")
	
	x = [0 for i in range(n)]
	for i in range(n - 1, -1, -1):
		x[i] = b[i]
		for j in range(i + 1, n):
			x[i] -= A[i][j] * x[j]
		x[i] /= A[i][i]

	print(x)

	for i in range(n):
		sum = 0
		for j in range(n):
			sum += CA[i][j] * x[j]
		print(sum)

	
n, A, b = read(sys.argv[1])
solve(n, A, b)