from PIL import Image, ImageDraw
import math

INPUT_FILE_NAME = 'data'

class Predicate:
	def __init__(self, threshold, name):
		self.threshold = threshold
		self.name = name
	def execute(self, val):
		pass 

class Eq(Predicate):
	def __init__(self, val, name):
		Predicate.__init__(self, val, name)
	def execute(self, val):
		return self.threshold == val

class Le(Predicate):
	def __init__(self, val, name):
		Predicate.__init__(self, val, name)
	def execute(self, val):
		return self.threshold <= val

class Ge(Predicate):
	def __init__(self, val, name):
		Predicate.__init__(self, val, name)
	def execute(self, val):
		return self.threshold >= val

class Node:
	def __init__(self, *args):
		if len(args) == 1:
			self.type = args[0]
			self.isLeaf = True
		else:
			self.false_branch = args[0]
			self.true_branch = args[1]
			self.predicate = args[2]
			self.isLeaf = False


class Creature:
	def __init__(self, bone_length, rotting_flesh, hair_length, has_soul, color):
		self.bone_length = bone_length
		self.rotting_flesh = rotting_flesh
		self.hair_length = hair_length
		self.has_soul = has_soul
		self.color = color

def entropy(y):
	types = set(y)
	entropy = 0
	for type in types:
		p = sum([1 if yt == type else 0 for yt in y]) * 1.0 / len(y)
		entropy -= p * math.log2(p)
	return entropy

def text_predicate(tree):
	if isinstance(tree.predicate, Eq):
		return tree.predicate.name + ' == ' + str(tree.predicate.threshold)
	elif isinstance(tree.predicate, Le):
		return tree.predicate.name + ' <= ' + str(tree.predicate.threshold)
	elif isinstance(tree.predicate, Ge):
		return tree.predicate.name + ' >= ' + str(tree.predicate.threshold)
	else:
		return 'ERROR'

class DecisionTree:
	
	def buildPredicates(self, X):
		B = []
		for x in X:
			B.append(Le(x.bone_length, 'bone_length'))
			B.append(Ge(x.bone_length, 'bone_length'))
			B.append(Le(x.rotting_flesh, 'rotting_flesh'))
			B.append(Ge(x.rotting_flesh, 'rotting_flesh'))
			B.append(Le(x.hair_length, 'hair_length'))
			B.append(Ge(x.hair_length, 'hair_length'))
			B.append(Le(x.has_soul, 'has_soul'))
			B.append(Ge(x.has_soul, 'has_soul'))
			B.append(Eq(x.color, 'color'))
		return B

	def build(self, X, y, score=entropy):
		B = self.buildPredicates(X)
		self.tree = self.learnID3(X, y, B, score)
		return self

	def learnID3(self, X, y, B, score):
		if len(set(y)) == 1:
			return Node(y[0])
		maxI = 0
		H = score(y)
		for b in B:
			I = H - self.countI(X, y, b, score)
			if I > maxI:
				maxI = I
				b_star = b
		X1, y1, X2, y2 = [], [], [], []
		for i in range(len(X)):
			if not self.execute(b_star, X[i]):
				X1.append(X[i])
				y1.append(y[i])
			else:
				X2.append(X[i])
				y2.append(y[i])
		if len(y1) == 0 or len(y2) == 0:
			return Node(self.majority(y))
		false_branch = self.learnID3(X1, y1, B, score)
		true_branch = self.learnID3(X2, y2, B, score)
		return Node(false_branch, true_branch, b_star)

	def majority(self, y):
		types = set(y)
		maxCnt = 0
		for type in types:
			cnt = sum([1 if yt == type else 0 for yt in y]) 
			if cnt > maxCnt:
				maxCnt = cnt
				res = type
		return res


	def countI(self, X, y, b, score):
		y1, y2 = [], []
		for i in range(len(X)):
			if self.execute(b, X[i]):
				y1.append(y[i])
			else:
				y2.append(y[i])
		return len(y1) * 1.0 / len(y) * score(y1) + len(y2) * 1.0 / len(y) * score(y2)

	def execute(self, b, x):
		if b.name == 'bone_length':
			return b.execute(x.bone_length)
		elif b.name == 'rotting_flesh':
			return b.execute(x.rotting_flesh)
		elif b.name == 'hair_length':
			return b.execute(x.hair_length)
		elif b.name == 'has_soul':
			return b.execute(x.has_soul)
		elif b.name == 'color':
			return b.execute(x.color)
		else:
			return False

	def predict(self, x):
		return self.dfs_predict(x, self.tree)

	def dfs_predict(self, x, node):
		if node.isLeaf:
			return node.type
		if self.execute(node.predicate, x):
			return self.dfs_predict(x, node.true_branch)
		else:
			return self.dfs_predict(x, node.false_branch)

	def important_predicates(self, type):
		lst = []
		self.dfs_important_predicates(self.tree, type, lst)
		return lst

	def dfs_important_predicates(self, node, type, ans):
		if node.isLeaf:
			return node.type == type
		left = self.dfs_important_predicates(node.false_branch, type, ans)
		right = self.dfs_important_predicates(node.true_branch, type, ans)
		if left or right:
			ans.append(text_predicate(node))
			return True
		return False


def read():
	lines = []
	with open(INPUT_FILE_NAME) as input_data_file:
		lines = input_data_file.readlines()
	X, y = [], []
	for line in lines:
		line = line[:len(line) - 1]
		attrs = line.split(",")
		X.append(Creature(float(attrs[0]), float(attrs[1]), float(attrs[2]), float(attrs[3]), attrs[4]))
		y.append(attrs[5])
	return X, y

def getwidth(tree):
	if tree.isLeaf:
		return 1
	return getwidth(tree.false_branch) + getwidth(tree.true_branch)

def getdepth(tree):
	if tree.isLeaf:
		return 1
	return 1 + max(getdepth(tree.false_branch), getdepth(tree.true_branch))

def drawtree(tree, path='tree.jpg'): 
	w = getwidth(tree) * 100
	h = getdepth(tree) * 100
	img = Image.new('RGB', (w, h), (255, 255, 255))
	draw = ImageDraw.Draw(img)
	drawnode(draw, tree, w / 2, 20)
	img.save(path, 'JPEG')

def drawnode(draw, tree, x, y): 
	if not tree.isLeaf:
		shift = 100
		width1 = getwidth(tree.false_branch) * shift
		width2 = getwidth(tree.true_branch) * shift
		left = x - (width1 + width2) / 2
		right = x + (width1 + width2) / 2
		predicate = text_predicate(tree)
		draw.text((x - 20, y - 10), predicate, (0, 0, 0))
		draw.line((x, y, left + width1 / 2, y + shift), fill=(255, 0, 0))
		draw.line((x, y, right - width2 / 2, y + shift), fill=(255, 0, 0))
		drawnode(draw, tree.false_branch, left + width1 / 2, y + shift)
		drawnode(draw, tree.true_branch, right - width2 / 2, y + shift)
	else:
		draw.text((x - 20, y), tree.type, (0, 0, 0))

X, y = read()
decisionTree = DecisionTree()
decisionTree.build(X, y)
drawtree(decisionTree.tree)
preds = decisionTree.important_predicates('Goblin')
print(preds)

		

