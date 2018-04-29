import math
import sys

def terminate(msg):
	print('program finished with error:')
	print(msg)
	exit()


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

def three_eights(X, Y, k, full_mode):
	n = len(X)
	fa, fb = Y[0], Y[-1]
	sx, sxx = 0, 0
	Is = []
	for iter in range(0, k):
		sx += sxx
		sxx = 0
		N = 3 ** iter
		h = (n - 1) // 3 // N
		for i in range(N):
			start = i * (n - 1) // N
			sxx += Y[start + h]
			sxx += Y[start + 2 * h]
		I = 3 * (X[h] - X[0]) / 8 * (fa + fb + 2 * sx + 3 * sxx)
		if full_mode:
			print('iter #%d, N=%d, h=%d, sx=%.5f, sxx=%.5f, I=%.5f' % (iter, N, h, sx, sxx, I))
		Is.append(I)
	return Is
		
def tr(X, Y):
	n = len(X)
	res = 0
	for i in range(n - 1):
		res += (Y[i + 1] + Y[i]) / 2 * (X[i + 1] - X[i])
	return res

def process_command_line_args():
	filename, full_mode = None, False
	for arg in sys.argv[1:]:
		if arg == '-f' or arg == '--full':
			full_mode = True
		elif arg.startswith('--input='):
			filename = arg[8:]
		else:
			terminate('unexpected %s arg was specified' % arg)
	if filename is None:
		terminate('specify file with input')
	return filename, full_mode

def check_n(n):
	k = 0
	while 3 ** k + 1 < n:
		k += 1
	if 3 ** k + 1 != n:
		terminate('for 3/8 method n should be 3**k + 1 for some k')
	return k


filename, full_mode = process_command_line_args()
n, X, Y = read(filename)
k = check_n(n)

if full_mode:
	I = tr(X, Y)
	print('integral by trapezoid method = %.5f' % I)

Is = three_eights(X, Y, k, full_mode)
if full_mode:
	for i in range(len(Is)):
		print('3/8, %d segments, I=%.5f' % (3 ** i, Is[i]))
print('integral by 3/8 = %.5f' % Is[-1])
for i in range(len(Is) - 1):
	ratio = 3 ** (len(Is) - 1 - i)
	print('ratio = %d, eps = %.5f' % (ratio, abs(Is[-1] - Is[i]) / (ratio ** 4 - 1)))



