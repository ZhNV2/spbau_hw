import matplotlib.pyplot as plt
import math

def f(x, y):
	return y / (x + 2) + x ** 2 + 2 * x

def y_t(x):
	return (1 + 0.5 * x ** 2) * (x + 2)

a, b = -1, 1
X_check = [-0.6, -0.3, 0, 0.3, 0.6]

for x_c in X_check:
	print('x = ', x_c)
	for i in range(7):
		


		n = 2 ** i
		h = (x_c - a) / n

		X = [a + h * i for i in range(n + 1)]
		Yt = [y_t(x) for x in X]
		Y = [0] * (n + 1)


		for d in [0, 0.015]:
			Y[0] = 1.5 + d
			for k in range(n):
				K1 = f(X[k], Y[k])
				K2 = f(X[k] + 1 / 3 * h, Y[k] + 1 / 3 * h * K1)
				K3 = f(X[k] + 2 / 3 * h, Y[k] + 2 / 3 * h * K2)
				Y[k + 1] = Y[k] + h * (1 / 4 * K1 + 3 / 4 * K3)
			print('n = ', n, ', eps = ', abs(Yt[n] - Y[n]), ', d = ', d)
	

	# to_show = n
	# plt.plot(X[:to_show], Yt[:to_show], color = 'blue') 
	# plt.plot(X[:to_show], Y[:to_show], color = 'orange')
	# plt.show()

