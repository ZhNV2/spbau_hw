from matplotlib.mlab import find
import numpy as np
import matplotlib.pyplot as pl
from cvxopt import spmatrix, matrix, sparse, solvers
from sklearn.datasets import make_blobs

def visualize(clf, X, y):
    border = .5
    h = .02

    x_min, x_max = X[:, 0].min() - border, X[:, 0].max() + border
    y_min, y_max = X[:, 1].min() - border, X[:, 1].max() + border

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    mesh = np.c_[xx.ravel(), yy.ravel()]

    z_class = clf.predict(mesh).reshape(xx.shape)

    # Put the result into a color plot
    pl.figure(1, figsize=(8, 6))
    pl.pcolormesh(xx, yy, z_class, cmap=pl.cm.summer, alpha=0.3)
    
    # Plot hyperplane and margin
    z_dist = clf.decision_function(mesh).reshape(xx.shape)
    pl.contour(xx, yy, z_dist, [0.0], colors='black')
    pl.contour(xx, yy, z_dist, [-1.0, 1.0], colors='black', linestyles='dashed')

    # Plot also the training points

    y_pred = clf.predict(X)
    
    ind_support = clf.support_
    ind_correct = list(set(find(y == y_pred)) - set(ind_support))
    ind_incorrect = list(set(find(y != y_pred)) - set(ind_support))
    
    pl.scatter(X[ind_correct, 0], X[ind_correct, 1], c=y[ind_correct], cmap=pl.cm.summer, alpha=0.9, edgecolors='black')
    pl.scatter(X[ind_incorrect, 0], X[ind_incorrect, 1], c=y[ind_incorrect], cmap=pl.cm.summer, alpha=0.9, marker='*', s=50, edgecolors='black')
    pl.scatter(X[ind_support, 0], X[ind_support, 1], c=y[ind_support], cmap=pl.cm.summer, alpha=0.9, linewidths=1.8, s=40, edgecolors='black')

    pl.xlim(xx.min(), xx.max())
    pl.ylim(yy.min(), yy.max())

class LinearSVM:
    def __init__(self, C=1):
        self.C = C
        self.EPS = 1e-6

    def fit(self, X, y):
        n = len(X[0])
        l = len(X)
        
        P = spmatrix(1, range(n), range(n), (n + 1 + l, n + 1 + l))
        q = matrix(spmatrix(self.C, range(n + 1, n + 1 + l), [0] * l, (n + 1 + l, 1)))
        h = matrix(spmatrix(-1, range(l, 2 * l), [0] * l, (2 * l, 1)))
        
        g00 = spmatrix([], [], [], (l, n))
        g01 = spmatrix([], [], [], (l, 1))
        g02 = spmatrix(-1, range(l), range(l))
        g10 = matrix((X.T * -y).T)
        g11 = matrix(-y, (l, 1))
        g12 = spmatrix(-1, range(l), range(l))
        G = sparse([[g00, g10], [g01, g11], [g02, g12]])

        qp_solution = np.array(solvers.qp(P, q, G, h)['x'])[:, 0]
        self.w = qp_solution[:n]
        self.w0 = qp_solution[n]
        ksi = qp_solution[n:]

        self.support_ = []
        for i in range(l):
            if abs((y[i] * self.decision_function(X[i])) - (1 - ksi[i])) < self.EPS:
                self.support_.append(i)

    def decision_function(self, X):
        return np.dot(X, self.w) + self.w0

    def predict(self, X):
        return np.sign(self.decision_function(X))

def linear_kernel(X1, x2):
    return np.dot(X1, x2)

def gauss_kernel(sigma):
    return lambda X1, x2: np.exp(-sigma * np.linalg.norm(X1 - x2, axis=1) ** 2) 

def polinomial_kernel(degree):
    return lambda X1, x2: (np.dot(X1, x2) + 1) ** degree

class KernelSVM:
    def __init__(self, C=1, kernel=None, sigma=1.0, degree=2):
        self.C = C
        self.EPS = 1e-6
        self.kernel = {
            'linear': linear_kernel,
            'gauss': gauss_kernel(sigma),
            'polinomial': polinomial_kernel(degree)
        }.get(kernel, linear_kernel)

    def fit(self, X, y):
        n = len(X[0])
        l = len(X)

        P = matrix(np.array([y * y[j] * self.kernel(X, X[j]) for j in range(l)]).transpose(), tc='d')
        q = matrix(-1, (l, 1), tc='d')
        G = sparse([spmatrix(1, range(l), range(l)), spmatrix(-1, range(l), range(l))])
        h = matrix(spmatrix(self.C, range(l), [0] * l, (2 * l, 1)))
        A = matrix(y, (1, l), tc='d')
        b = matrix([0], tc='d')

        alphas = np.array(solvers.qp(P, q, G, h, A, b)['x'])[:, 0]
        alp, support_, sup_y, sup_X = [], [], [], []
        for i in range(l):  
            if alphas[i] > self.EPS:
                alp.append(alphas[i])
                support_.append(i)
                sup_y.append(y[i])
                sup_X.append(np.array(X[i]))
        self.alp, self.support_, self.sup_y, self.sup_X = np.array(alp), np.array(support_), np.array(sup_y), np.array(sup_X)
        self.w0 = np.mean([sup_y - sum([alphas[j] * y[j] * self.kernel(sup_X, X[j]) for j in range(l)])])

    def decision_function(self, X):
        res = sum([self.kernel(X, self.sup_X[i]) * self.alp[i] * self.sup_y[i] for i in range(len(self.sup_X))])
        return res
        
    def predict(self, X):
        return np.sign(self.decision_function(X))
 

def test(svm, X, y, file):
    svm.fit(X, y)
    visualize(svm, X, y)
    pl.savefig(file)
    pl.close()

def gen():
    X, y = make_blobs(n_samples=100, centers=2, n_features=2, cluster_std=4)
    y = 2 * y - 1
    return X, y


X, y = gen()
test(LinearSVM(), X, y, 'linear')
test(KernelSVM(), X, y, 'kernel')

sigma = [0.25, 0.5, 1, 2, 3]
for sig in sigma:
    test(KernelSVM(kernel='gauss', sigma=sig), X, y, 'kernel_gauss_sig=%.2f.png' % sig)

# degrees = [1, 2, 3, 4, 5]
# for deg in degrees:
#     test(KernelSVM(kernel='polinomial', degree=deg), X, y, 'kernel_poly_deg=%d.png' % deg)

