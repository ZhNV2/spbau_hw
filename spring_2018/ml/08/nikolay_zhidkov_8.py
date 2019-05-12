from matplotlib import pyplot as plt
import numpy as np
import time
import random

FILE_NAME = 'boston.csv'
SPLIT_RATIO = 0.8
T = 0

def train_test_split(X, y, ratio):
	p = np.random.permutation(len(y))
	X, y = X[p], y[p]
	train_size = int(len(y) * ratio)
	return X[:train_size], y[:train_size], X[train_size:], y[train_size:]

def read_data(filename):
	data = np.genfromtxt(filename, comments="#", delimiter=",")
	X, y = [], []
	for obj in data:
		X.append(np.array(obj[:len(obj) - 1]))
		y.append(obj[-1])
	return np.array(X), np.array(y)

def sample(size, weights):
	X = np.ones((size, 2))
	X[:, 1] = np.random.gamma(4., 2., size) 
	y = X.dot(np.asarray(weights))
	y += np.random.normal(0, 25, size) 
	return X[:, 1:], y

def visualize(size, lr):
	X, y_true = sample(size, weights=[24., 42.])
	lr.fit(X, y_true)
	plt.scatter(X, y_true)
	plt.plot(X, lr.predict(X), color="red")
	plt.show()

def mse(y_test, y_pred):
	return np.mean(np.square(y_test - y_pred))

class NormalLR:

	def __init__(self, t):
		self.t = t

	def fit(self, X, y):
		self.weights = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + self.t * np.identity(X.shape[1])), X.T), y)
		return self

	def predict(self, X):
		return np.dot(X, self.weights)

class GradientLR(NormalLR):
	def __init__(self, alpha, t):
		if alpha <= 0:
			raise ValueError("alpha should be positive")
		self.t = t
		self.alpha = alpha
		self.threshold = alpha / 100

	def fit(self, X, y): 
		l, n = X.shape
		self.weights = np.random.uniform(-0.5 * n, 0.5 * n, n) 
		while True:
			old_weights = self.weights
			self.weights = old_weights - self.alpha / l * (np.dot(X.T, np.dot(X, old_weights) - y) + 2 * self.t * old_weights)
			if np.linalg.norm(old_weights - self.weights) < self.threshold:
				break
		return self

def get_time_ms():
	return int(round(time.time() * 1000))

def run(lr, X_test, y_test, X_pred, y_pred):
	lr.fit(X_test, y_test)
	y = lr.predict(X_pred)
	return mse(y_pred, y)

def compare(normal_lr, gradient_lr, X_test, y_test, X_pred, y_pred, size=None):
	t1 = get_time_ms()
	normal_q = run(normal_lr, X_test, y_test, X_pred, y_pred)
	t2 = get_time_ms()
	gradient_q = run(gradient_lr, X_test, y_test, X_pred, y_pred)
	t3 = get_time_ms()
	if size:
		print('size=%d' % size)
	print('norm, q = %.10f, time = %d ms' % (normal_q, t2 - t1))
	print('grad, q = %.10f, time = %d ms' % (gradient_q, t3 - t2))
	print('norm_q - grad_q = %.10f' % (normal_q - gradient_q))

def test_sample():
	normal_lr = NormalLR()
	gradient_lr = GradientLR(0.01)
	sizes = [2 ** i for i in range(7, 20)]
	for size in sizes:
		X, y_true = sample(size, weights=[24., 42.])
		compare(normal_lr, gradient_lr, X, y_true, X, y_true, size)

def normalize(X, y):
	n = X.shape[1]
	mins = np.amin(X, axis=0)
	maxs = np.amax(X, axis=0)
	for x in X:
		for i in range(n):
			x[i] = (x[i] - mins[i]) / (maxs[i] - mins[i])
	min_y = np.min(y)
	max_y = np.max(y)
	for i in range(len(y)):
		y[i] = (y[i] - min_y) / (max_y - min_y)


def test_real_data():
	X, y = read_data(FILE_NAME)
	normalize(X, y)
	X_test, y_test, X_pred, y_pred = train_test_split(X, y, SPLIT_RATIO)
	normal_lr = NormalLR(T)
	gradient_lr = GradientLR(0.01, T)
	compare(normal_lr, gradient_lr, X_test, y_test, X_pred, y_pred)

	print('norm w:', normal_lr.weights)


#test_sample()
test_real_data()