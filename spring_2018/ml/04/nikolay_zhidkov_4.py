import re
import math

INPUT_FILE_NAME = 'data'

def read():
	lines = []
	with open(INPUT_FILE_NAME) as input_data_file:
		lines = input_data_file.readlines()
	X, y = [], []
	for line in lines:
		tup = line.split('am', 1)
		X.append(tup[1])
		y.append(tup[0] + 'am')
	return X, y

class NaiveBayes:
	def __init__(self, alpha):
		self.alpha = alpha

	def vectorize(self, strs):
		splitted_strs = []
		words = set()
		for s in strs:
			splitted_strs.append(re.split('\W+', s))
			words.update(splitted_strs[-1])
		self.list_of_words = list(words)
		self.V = len(self.list_of_words)
		self.word_to_pos = {}
		for i in range(self.V):
			self.word_to_pos[self.list_of_words[i]] = i
		bag = []
		for s in splitted_strs:
			bag.append([0] * len(self.list_of_words))
			for word in s:
				if len(word) != 0:
					bag[-1][self.word_to_pos[word]] += 1
		return bag

	def fit(self, X, y):
		self.classes = set(y)
		self.bag = self.vectorize(X)
		words_cnt = {}
		for cl in self.classes:
			words_cnt[cl] = 0
			for i in range(len(y)):
				if y[i] == cl:
					words_cnt[cl] += sum(self.bag[i])
		self.freqs = {}
		for i in range(len(y)):
			for pos in range(self.V):
				self.freqs[(y[i], self.list_of_words[pos])] = self.freqs.get((y[i], self.list_of_words[pos]), 0) + self.bag[i][pos]
		for cl in self.classes:
			for word in self.list_of_words:
				self.freqs[(cl, word)] = 1.0 * (self.freqs.get((cl, word), 0) + self.alpha) / (words_cnt[cl] + 1.0 * self.alpha * self.V)
		self.P = {}
		for cl in self.classes:
			self.P[cl] = 1.0 * sum([1 if y[i] == cl else 0 for i in range(len(y))]) / len(y)
			self.P[cl] = math.log(self.P[cl])

	def ber(self, x, tetha):
		return (tetha ** x) * ((1 - tetha) ** (1 - x))

	def predict(self, X):
		s = []
		for x in X:
			ans, ans_y = None, None
			xd = [0] * self.V
			words = re.split('\W+', x)
			for word in words:
				if word in self.word_to_pos:
					xd[self.word_to_pos[word]] += 1
			for cl in self.classes:
				res = self.P[cl]
				for i in range(self.V):
					res += math.log(self.ber(xd[i], self.freqs.get((cl, self.list_of_words[i]), 0)))
				if ans is None or ans < res:
					ans = res
					ans_y = cl
			s.append(ans_y)
		return s


	def score(self, X, y):
		s = self.predict(X)
		success = sum([1 if s[i] == y[i] else 0 for i in range(len(y))])
		return 1.0 * success / len(y)

X, y = read()
n = int(len(y) * 0.8)
X_1, y_1 = X[:n], y[:n]
X_2, y_2 = X[n:], y[n:]

bayes = NaiveBayes(1)
bayes.fit(X_1, y_1)
score = bayes.score(X_2, y_2)

print(score)
