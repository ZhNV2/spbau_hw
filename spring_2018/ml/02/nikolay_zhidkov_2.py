import cv2
import random
import numpy
import sys

HIST_H = 50
HIST_W = 200
COLORS = 16

def read_image(path): 
	image = cv2.imread(path)
	return [[[int(x[0]), int(x[1]), int(x[2])] for x in row] for row in image]

def nearest_centr_index(x, centrs, distance_metric):
	idx = 0
	min_dist = distance_metric(x, centrs[0])
	for i in range(len(centrs)):
		dist_to_centr = distance_metric(x, centrs[i])
		if min_dist > dist_to_centr:
			min_dist = dist_to_centr
			idx = i
	return idx

def get_average(X):
	if len(X) == 0:
		return [0, 0, 0]
	return [sum([X[j][i] for j in range(len(X))]) // len(X) for i in range(len(X[0]))]

def k_means(X, n_clusters, distance_metric):
	Xes = [x for row in X for x in row]
	n = len(Xes)
	centrs = []
	for i in range(n_clusters):
		new_centr = Xes[random.randint(0, n - 1)]
		while new_centr in centrs:
			new_centr = Xes[random.randint(0, n - 1)]
		centrs.append(new_centr)
	print('init centrs')
	print(centrs)
	while True:
		members = [[] for _ in range(n_clusters)]
		for x in Xes:
			members[nearest_centr_index(x, centrs, distance_metric)].append(x)
		new_centrs = [get_average(cluster) for cluster in members]
		print(new_centrs)
		if centrs == new_centrs:
			break
		centrs = new_centrs
	labels = [[centrs[nearest_centr_index(x, centrs, distance_metric)] for x in row] for row in X]
	return labels, centrs

def centroid_histogram(labels, centrs):
	hist = []
	list_labels = [label for row in labels for label in row]
	for centr in centrs:
		hist.append(100 * sum([1 if centr == x_centr else 0 for x_centr in list_labels]) // len(list_labels))
	return hist

def plot_colors(hist, centrs):
	bar = numpy.zeros((HIST_H, HIST_W, 3), numpy.uint8)
	start_x = 0
	for (percent, color) in zip(hist, centrs):
		end_x = start_x + percent * HIST_W / float(100)
		cv2.rectangle(bar, (int(start_x), 0), (int(end_x), 50), color, -1)
		start_x = end_x
	cv2.rectangle(bar, (int(start_x), 0), (int(HIST_W), 50), centrs[-1], -1)
	return bar

def euclid_dist(x, y):
	return sum([(x[i] - y[i]) ** 2 for i in range(len(x))]) ** 0.5

def recolor(image, n_colors):
	labels, centrs = k_means(image, n_colors, euclid_dist)
	hist = centroid_histogram(labels, centrs)
	bar = plot_colors(hist, centrs)
	return bar, labels


def main():
	path = sys.argv[1]
	image = read_image(path)
	bar, labels = recolor(image, COLORS)
	cv2.imwrite('hist.png', bar)
	new_image = numpy.zeros((len(image), len(image[0]), 3))
	for i in range(len(new_image)):
		for j in range(len((new_image[i]))):
			new_image[i][j][0], new_image[i][j][1], new_image[i][j][2] = labels[i][j][0], labels[i][j][1], labels[i][j][2]
	cv2.imwrite('new_image.png', new_image)

main()






