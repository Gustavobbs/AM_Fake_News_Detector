import numpy as np
import sklearn.metrics.pairwise

def distance(x, X):

    m = X.shape[0]
    D = np.zeros(m)

    D = sklearn.metrics.pairwise.pairwise_distances(X=X, Y=x, metric='euclidean')

    return D

def knn(x, X, Y, K):

    y = 0
    ind_viz = np.ones(K, dtype=int)

    dist = distance(x, X)[:,0]
    idx = np.argsort(dist, kind='mergesort')
    ind_viz = idx[0:K]
    y = np.argmax(np.bincount(Y[ind_viz]))

    classes, countClasses = np.unique(Y[ind_viz], return_counts=True)

    return y

def testSamples(Xtrain, Ytrain, Xtest, K):

    Ypred = [knn(x, Xtrain, Ytrain, K) for x in Xtest]

    return Ypred
