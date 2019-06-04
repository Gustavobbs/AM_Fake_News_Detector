import numpy as np
import scipy

def distance(x, X):

    m = X.shape[0]
    D = np.zeros(m)

    x2 = x.copy()
    x2.data **= 2
    x2 = np.array(x2.sum(axis=1))
    D = (np.sqrt(X.multiply(X).sum(axis=1).T - (2 * x.dot(X.transpose()).toarray()) + x2)).tolist()[0]

    return D

def knn(x, X, Y, K):

    y = 0
    ind_viz = np.ones(K, dtype=int)

    dist = distance(x, X)
    idx = np.argsort(dist, kind='mergesort')
    ind_viz = idx[0:K]
    y = np.argmax(np.bincount(Y[ind_viz]))

    classes, countClasses = np.unique(Y[ind_viz], return_counts=True)

    return y

def testSamples(Xtrain, Ytrain, Xtest, K):

    Ypred = [knn(x, Xtrain, Ytrain, K) for x in Xtest]

    return Ypred
