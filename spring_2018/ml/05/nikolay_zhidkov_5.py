import sys
import numpy
import matplotlib.pyplot as plt

GRAD_MAX_ITERS = 100
SPLIT_RATIO = 0.9
ALP = [1e-6, 1e-4, 1e-2, 1]
K = [1, 10, 50]

def read_data(filename):
	data = numpy.genfromtxt(filename, comments="#", delimiter=",")
	X, y = [], []
	for obj in data:
		X.append(numpy.array(obj[:len(obj) - 1]))
		y.append(1 if obj[-1] == 1 else -1)
	return numpy.array(X), numpy.array(y)

def train_test_split(X, y, ratio):
	p = numpy.random.permutation(len(y))
	X, y = X[p], y[p]
	train_size = int(len(y) * ratio)
	return X[:train_size], y[:train_size], X[train_size:], y[train_size:]

def print_precision_recall(y_pred, y_test):
	classes = set(y_test)
	for c in classes:
		tp = sum([1 if y_pred[i] == c and y_test[i] == c else 0 for i in range(len(y_pred))])
		fp = sum([1 if y_pred[i] == c and y_test[i] != c else 0 for i in range(len(y_pred))])
		fn = sum([1 if y_pred[i] != y_test[i] and y_test[i] != c else 0 for i in range(len(y_pred))])
		precision = tp / (tp + fp + 0.)
		recall = tp / (tp + fn + 0.)
		print(c, precision, recall)

def log_loss(M):
	return numpy.log2(1 + numpy.exp(-M)), -1. / numpy.log(2) / (1 + numpy.exp(M))

def sigmoid_loss(M):
	return 2. / (1 + numpy.exp(M)), -2 * numpy.exp(M) / (1 + numpy.exp(M)) ** 2

def count_loss(X, y, w, loss):
	return loss(numpy.dot(X, w) * y)

def gen_weights(n):
	return numpy.random.rand(n) * (1. / n) - 0.5 / n

def normalize(w):
	w /= numpy.linalg.norm(w)

class GradientDescent:
	def __init__(self, alpha, threshold=1e-5, loss=log_loss):
		if alpha <= 0:
			raise ValueError("alpha should be positive")
		if threshold <= 0:
			raise ValueError("threshold should be positive")
		self.alpha = alpha
		self.threshold = threshold
		self.loss = loss

	def fit(self, X, y):
		errors = []
		w = gen_weights(len(X[0]))
		normalize(w)
		for _ in range(GRAD_MAX_ITERS):
			l, dl = count_loss(X, y, w, self.loss)
			errors.append(sum(l))
			dq = sum(X * (dl * y)[:, numpy.newaxis])
			w, prev_w = w - self.alpha * dq, w
			normalize(w)
			if numpy.linalg.norm(w - prev_w) < self.threshold:
				break
		self.weights = w
		return errors

	def predict(self, X):
		return numpy.array(list(map(int, numpy.sign(numpy.dot(X, self.weights)))))

class SGD:
	def __init__(self, alpha, loss=log_loss, k=1, n_iter=5000):
		if alpha <= 0:
			raise ValueError("alpha should be positive")
		if k <= 0 or not isinstance(k, int):
			return ValueError("k should be a positive integer")
		if n_iter <= 0 or not isinstance(n_iter, int):
			raise ValueError("n_iter should be a positive integer")
		self.k = k
		self.n_iter = n_iter
		self.alpha = alpha
		self.loss = loss

	def fit(self, X, y):
		errors = []
		w = gen_weights(len(X[0]))
		normalize(w)
		l, dl = count_loss(X, y, w, self.loss)
		Q = sum(l)
		n = 1. / len(y)
		for _ in range(self.n_iter):
			sub = numpy.random.choice(len(y), self.k)
			sub_X, sub_y = X[sub], y[sub]
			l, dl = count_loss(sub_X, sub_y, w, self.loss)
			dq = sum(sub_X * (dl * sub_y)[:, numpy.newaxis])
			eps = sum(l)
			w -= self.alpha * dq
			normalize(w)
			Q = (1 - n) * Q + n * eps
			errors.append(Q)
		self.weights = w
		return errors

	def predict(self, X):
		return numpy.sign(numpy.dot(X, self.weights))



def test_GradientDescent(X_train, y_train, X_test, y_test, alphas, losses):
	for loss in losses:
		for alp in alphas:
			gd = GradientDescent(alpha=alp, loss=loss)
			errors = gd.fit(X_train, y_train)
			y_pred = gd.predict(X_train)
			print('GradientDescent with loss=%s, alp = %.6f' % (loss.__name__, alp))
			print_precision_recall(y_pred, y_train)
			plt.plot(numpy.arange(len(errors)), errors, label='alp=%.6f' % alp)
		plt.xlabel('iters')
		plt.ylabel('Q')
		plt.title('grad, %s' % loss.__name__)
		plt.legend()
		plt.savefig('grad,%s' % loss.__name__)
		plt.close()

def test_SGD(X_train, y_train, X_test, y_test, alphas, losses, ks):
	for loss in losses:
		for i in range(len(ks)):
			k = ks[i]
			for alp in alphas:
				gd = SGD(alpha=alp, loss=loss, k=k)
				errors = gd.fit(X_train, y_train)
				y_pred = gd.predict(X_train)
				print('SGD with loss=%s, alp = %.6f, k=%d' % (loss.__name__, alp, k))
				print_precision_recall(y_pred, y_train)
				f = plt.subplot(1, len(ks), i + 1)
				f.set_title('k=%d' % k)
				plt.plot(numpy.arange(len(errors)), errors, label='alp=%.6f' % alp)
				plt.xlabel('iters')
				plt.ylabel('Q')
		plt.suptitle('sgd, %s' % loss.__name__)
		plt.legend()
		plt.savefig('sgd,%s' % loss.__name__)
		plt.close()






def main():
	if len(sys.argv) < 2:
		print('usage: python3 %s filename.csv' % sys.argv[0])
		exit(1)
	filename = sys.argv[1]
	X, y = read_data(filename)
	for i in range(len(X[0])):
		X[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()		
	X_train, y_train, X_test, y_test = train_test_split(X, y, SPLIT_RATIO)
	test_GradientDescent(X_train, y_train, X_test, y_test, ALP, [sigmoid_loss, log_loss])
	test_SGD(X_train, y_train, X_test, y_test, ALP, [sigmoid_loss, log_loss], K)


main()