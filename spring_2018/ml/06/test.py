import scipy.io
from sklearn.cross_validation import train_test_split
import math
import random

def get_data():
	dataset = scipy.io.loadmat('mnist-original.mat')
	trainX, testX, trainY, testY = train_test_split(
		dataset['data'].T / 255.0, dataset['label'].squeeze().astype("int0"), test_size = 0.3)
	return trainX, testX, trainY, testY

def sigmoid(x):
	return 2. / (1 + math.exp(x))

def sigmoid_derivative(x):
	return -2 * math.exp(x) / (1 + math.exp(x)) ** 2

class NeuralNetwork:
	def __init__(self, layers):
		self.num_layers = len(layers)
		self.layers = layers

		self.output_value, self.eps, self.w, self.output_derivative = [], [], [], []
		for i in range(self.num_layers):
			layer_sz = self.layers[i]
			self.output_value.append([0 for _ in range(layer_sz)])
			self.output_derivative.append([0 for _ in range(layer_sz)])
			self.eps.append([0 for _ in range(layer_sz)])
			if i == 0:
				self.w.append([])
			else:
				self.w.append([[random.random() - 0.5 for _ in range(self.layers[i - 1] + 1)] for _ in range(layer_sz)])

	def train(self, X, y, max_iter=10000, learning_rate=1e-1):
		for j in range(max_iter):
			if j % 1000 == 0:
				print(j)
			index = random.randint(0, len(y) - 1)
			self.forward(X[index], y[index])
			self.backward(learning_rate)


	def forward(self, x, y):
		y = [1 if i == y else 0 for i in range(10)]
		for i in range(len(x)):
			self.output_value[0][i] = x[i]
		for i in range(1, self.num_layers):
			layer_sz = self.layers[i]
			prev_layer_sz = self.layers[i - 1]
			for j in range(layer_sz):
				value = -self.w[i][j][prev_layer_sz];
				for k in range(prev_layer_sz):
					value += self.w[i][j][k] * self.output_value[i - 1][k]
				self.output_value[i][j] = sigmoid(value)
				self.output_derivative[i][j] = sigmoid_derivative(value)
		for i in range(self.layers[-1]):
			self.eps[self.num_layers - 1][i] = self.output_value[self.num_layers - 1][i] - y[i]

	def backward(self, learning_rate):
		for i in range(self.num_layers - 2, -1, -1):
			layer_sz = self.layers[i]
			next_layer_sz = self.layers[i + 1]
			for j in range(layer_sz):
				self.eps[i][j] = 0
				for k in range(next_layer_sz):
					self.eps[i][j] += self.eps[i + 1][k] * self.output_derivative[i + 1][k] * self.w[i + 1][k][j]
		for i in range(self.num_layers - 1, 0, -1):
			layer_sz = self.layers[i]
			prev_layer_sz = self.layers[i - 1]
			for j in range(layer_sz):
				for k in range(prev_layer_sz):
					#print(abs(learning_rate * self.eps[i][j] * self.output_derivative[i][j] * self.output_value[i - 1][k]))
					self.w[i][j][k] -= learning_rate * self.eps[i][j] * self.output_derivative[i][j] * self.output_value[i - 1][k]
				self.w[i][j][prev_layer_sz] -= learning_rate * self.eps[i][j] * self.output_derivative[i][j]

	def predict(self, X): 
		y = []
		print(self.w[1])
		for i in range(len(X)):
			if i % 1000 == 0:
				print(i)
			x = X[i]
			self.forward(x, 0)
			maxP = -1
			for i in range(len(self.output_value[-1])):
				if maxP == -1 or self.output_value[-1][i] > self.output_value[-1][maxP]:
					maxP = i
			y.append(maxP)
		return y

trainX, testX, trainY, testY = get_data()
nn = NeuralNetwork([trainX.shape[1], 10])
nn.train(trainX, trainY, max_iter=2*len(trainX))
predY = nn.predict(testX)
res = 1.0 * sum([1 if predY[i] == testY[i] else 0 for i in range(len(testY))]) / len(testY)
print(res)


