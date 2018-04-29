import scipy.io
from sklearn.cross_validation import train_test_split
import random
import numpy

def get_data():
	dataset = scipy.io.loadmat('mnist-original.mat')
	trainX, testX, trainY, testY = train_test_split(
		dataset['data'].T / 255.0, dataset['label'].squeeze().astype("int0"), test_size = 0.3)
	return trainX, testX, trainY, testY

def sigmoid_loss(M):
	return 2. / (1 + numpy.exp(M)), -2 * numpy.exp(M) / (1 + numpy.exp(M)) ** 2

class NeuralNetwork:
	def __init__(self, layers):
		self.num_layers = len(layers)
		self.layers = layers

	def init_network(self):
		self.output_value, self.eps, self.w, self.output_derivative = [], [], [], []
		for i in range(self.num_layers):
			layer_sz = self.layers[i]
			self.output_value.append(numpy.concatenate([numpy.zeros(layer_sz), [-1]]))
			self.output_derivative.append(numpy.concatenate([numpy.zeros(layer_sz), [-1]]))
			self.eps.append(numpy.zeros(layer_sz + 1))
			if i == 0:
				self.w.append([])
			else:
				prev_layer_sz = self.layers[i - 1] 
				self.w.append(numpy.random.rand(layer_sz, prev_layer_sz + 1) - 0.5 * numpy.ones((layer_sz, prev_layer_sz + 1)))
				
	def train(self, X, y, max_iter=100000, learning_rate=1e-1):
		self.init_network()
		for j in range(max_iter):
			index = random.randint(0, len(y) - 1)
			self.forward(X[index], y[index])
			self.backward(learning_rate)

	def forward(self, x, y):
		y = numpy.array([1 if i == y else 0 for i in range(10)])
		for i in range(len(x)):
			self.output_value[0][i] = x[i]
		for i in range(1, self.num_layers):
			input_values = numpy.dot(self.w[i], self.output_value[i - 1])
			self.output_value[i], self.output_derivative[i] = sigmoid_loss(input_values)
		self.eps[-1] = self.output_value[-1] - y
		
	def backward(self, learning_rate):
		for i in range(self.num_layers - 2, -1, -1):
			self.eps[i] = numpy.dot(numpy.multiply(self.eps[i + 1], self.output_derivative[i + 1]), self.w[i + 1])

		for i in range(self.num_layers - 1, 0, -1):
			self.w[i] -= learning_rate * numpy.outer(numpy.multiply(self.eps[i], self.output_derivative[i]), self.output_value[i - 1])

	def predict(self, X): 
		y = []
		for i in range(len(X)):
			x = X[i]
			self.forward(x, 0)
			maxP = -1
			for i in range(len(self.output_value[-1])):
				if maxP == -1 or self.output_value[-1][i] > self.output_value[-1][maxP]:
					maxP = i
			y.append(maxP)
		return y


def test(layers):
	nn = NeuralNetwork([trainX.shape[1], 10])
	nn.train(trainX, trainY)
	predY = nn.predict(testX)
	res = 1.0 * sum([1 if predY[i] == testY[i] else 0 for i in range(len(testY))]) / len(testY)
	print('layers = ', layers, ', res = ', res * 100 , '%')


trainX, testX, trainY, testY = get_data()
print('2 layers')
test([trainX.shape[1], 10])
print('3 layers')
test([trainX.shape[1], 10, 10])
test([trainX.shape[1], 25, 10])
test([trainX.shape[1], 50, 10])
test([trainX.shape[1], 100, 10])
test([trainX.shape[1], 500, 10])
test('4 layers')
test([trainX.shape[1], 10, 10, 10])
test([trainX.shape[1], 100, 10, 10])
test([trainX.shape[1], 100, 100, 10])
test([trainX.shape[1], 10, 100, 10])
print('5 layers')
test([trainX.shape[1], 10, 10, 10, 10])
test([trainX.shape[1], 100, 50, 10, 10])
test([trainX.shape[1], 100, 50, 50, 10])
test([trainX.shape[1], 100, 100, 10, 10])
test([trainX.shape[1], 100, 100, 50, 10])


