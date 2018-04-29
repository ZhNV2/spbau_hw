import math
import sys

def f(x):
	return 1 / (1 + x ** 2)

def antiderivative(x):
	return math.atan(x)

def integral(a, b):
	return antiderivative(b) - antiderivative(a)

def get(t, a, b):
	return (a + b) / 2 + (b - a) / 2 * t

def three_eighths_eps(f, integral, a, b, eps):
	fa = f(a)
	fb = f(b)
	tI = integral(a, b)
	print(tI)
	sx, sxx = 0, 0
	I = 10e9
	iter = 0
	okR, okD, okL = False, False, False
	x = [-1, -1 / (5 ** 0.5), 1 / (5 ** 0.5), 1]
	A = [1 / 6, 5 / 6, 5 / 6, 1 / 6]
	while not okR or not okD or not okL:
		sx += sxx
		sxx = 0
		N = 3 ** iter
		h = (b - a) / N / 3
		lab = 0
		for i in range(N):
			start = a + i * (b - a) / N
			sxx += f(start + h)
			sxx += f(start + 2 * h)
			for j in range(4):
				lab += A[j] * f(get(x[j], start, start + 3 * h))
		lab *= 3 * h / 2
		newI = 3 * h / 8 * (fa + fb + 2 * sx + 3 * sxx)
		if not okR and abs(I - newI) / (3 ** 4 - 1) < eps:
			print('runge finised within %d iters, I = %.5f, delta = %.5f' % (iter, newI, abs(newI - tI)))
			okR = True
		if not okD and abs(newI - tI) < eps:
			print('delta finised within %d iters, I = %.5f, delta = %.5f' % (iter, newI, abs(newI - tI)))
			okD = True
		if not okL and abs(lab - tI) < eps:
			print('delta finised within %d iters, I = %.5f, delta = %.5f' % (iter, lab, abs(lab - tI)))
			okL = True
		iter += 1
		I = newI



eps = float(sys.argv[1])
three_eighths_eps(f, integral, 0, 100, eps)