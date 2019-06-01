import numpy as np

def distance(x, X):

    m = X.shape[0]
    D = np.zeros(m)

    for i in range(m):
        D[i] = np.linalg.norm(X[i]-x)

    return D

def knn(x, X, Y, K):

    y = 0
    ind_viz = np.ones(K, dtype=int)

    dist = distance(x, X)
    idx = np.argsort(dist, kind='mergesort')
    ind_viz = idx[0:K]
    y = np.argmax(np.bincount(Y[ind_viz]))

    classes, countClasses = np.unique(Y[ind_viz], return_counts=True)
    print(classes)
    print(np.argmax(countClasses))

    return y, ind_viz

def testSamples(Xtrain, Ytrain, Xtest, K):

    Ypred = [knn(x, Xtrain, Ytrain, K) for x in Xtest]

    return Ypred
