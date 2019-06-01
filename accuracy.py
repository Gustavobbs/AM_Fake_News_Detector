import numpy as np

def accuracy(Ypred, Y):

    return np.sum(Ypred == Y)/ len(Y)
