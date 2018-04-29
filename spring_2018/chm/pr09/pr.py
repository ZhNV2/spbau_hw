import matplotlib.pyplot as plt
import math

def f(x, y):
	return 2 * x * (x ** 2 + y)

def y_t(x):
	return math.exp(x ** 2) * 3 - x * x - 1

a, b = 0, 6
n = 600
h = (b - a) / n

X = [a + h * i for i in range(n + 1)]
Yt = [y_t(x) for x in X]
Y = [0] * (n + 1)


Y[0] = 2
for k in range(n):
	K1 = f(X[k], Y[k])
	K2 = f(X[k] + 1 / 3 * h, Y[k] + 1 / 3 * h * K1)
	K3 = f(X[k] + 2 / 3 * h, Y[k] + 2 / 3 * h * K2)
	Y[k + 1] = Y[k] + h * (1 / 4 * K1 + 3 / 4 * K3)

to_show = n
plt.plot(X[:to_show], Yt[:to_show], color = 'blue') 
plt.plot(X[:to_show], Y[:to_show], color = 'orange')
plt.show()

