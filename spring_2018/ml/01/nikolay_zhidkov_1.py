import random

INPUT_FILE_NAME = 'data'
SPLIT_RATIO = 0.66

def train_test_split(X, y, ratio):
	combined = zip(X, y)
	random.shuffle(combined)
	X, y = zip(*combined)
	train_size = int(len(X) * ratio)
	return X[:train_size], y[:train_size], X[train_size:], y[train_size:]

def most_common(lst):
    return max(set(lst), key=lst.count)

def euclid_dist(u, v):
	return sum([(u[i] - v[i]) ** 2 for i in range(len(u))]) ** 0.5

def manhattan_dist(u, v):
	return sum([abs(u[i] - v[i]) for i in range(len(u))])

def knn(X_train, y_train, X_test, k, dist):
	y_test = []
	for u in X_test:
		k_nearest = sorted(zip(X_train, y_train), key=lambda x: dist(u, x[0]))[:k]
		k_nearest_classes = [x[1] for x in k_nearest]
		y_test.append(most_common(k_nearest_classes))
	return y_test

def print_precision_recall(y_pred, y_test):
	classes = set(y_test)
	for c in classes:
		tp = sum([1 if y_pred[i] == c and y_test[i] == c else 0 for i in range(len(y_pred))])
		fp = sum([1 if y_pred[i] == c and y_test[i] != c else 0 for i in range(len(y_pred))])
		fn = sum([1 if y_pred[i] != y_test[i] and y_test[i] != c else 0 for i in range(len(y_pred))])
		precision = tp / (tp + fp + 0.)
		recall = tp / (tp + fn + 0.)
		print(c, precision, recall)

def knn_for_one(X_train, y_train, k, dist, pos):
	X_train_ = X_train[:pos] + X_train[pos + 1:]
	y_train_ = y_train[:pos] + y_train[pos + 1:]
	return knn(X_train_, y_train_, [X_train[pos]], k, dist)

def loocv(X_train, y_train, dist):
	return min(range(1, len(X_train) - 1), key=lambda k:
		sum([knn_for_one(X_train, y_train, k, dist, i)[0] != y_train[i] for i in range(len(X_train))]))

def precision_recall(X_train, y_train, X_test, y_test, dist):
	opt_k = loocv(X_train, y_train, dist)
	y_pred = knn(X_train, y_train, X_test, opt_k, dist)
	print_precision_recall(y_pred, y_test)

def main():
	X, y = [], []
	with open(INPUT_FILE_NAME) as input_data_file:
		lines = input_data_file.readlines()
		for line in lines:
			data = map(float, line.split(','))
			y.append(int(data[0]))
			X.append(data[1:])
	X_train, y_train, X_test, y_test = train_test_split(X, y, SPLIT_RATIO)

	print('euclid_dist')
	precision_recall(X_train, y_train, X_test, y_test, euclid_dist)
	print('manhattan_dist')
	precision_recall(X_train, y_train, X_test, y_test, manhattan_dist)
	
main()
