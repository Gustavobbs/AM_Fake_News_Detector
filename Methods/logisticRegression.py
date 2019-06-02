import numpy as np
import pandas as pd
import scipy
import scipy.optimize
from scipy import sparse

def sigmoid(z):

    return 1 / (1 + np.exp(-z))

def costReg(theta, X, Y, lambda_reg):

    m = Y.shape[0]
    J = 0
    grad = np.zeros( theta.shape[0] )
    eps = 1e-15

    h = sigmoid((X.multiply(theta)).sum(axis=1).getA()[:, 0])
    h1 = [eps if i == 0 else i for i in h]
    h2 = [eps if (1 - i) == 0 else (1 - i) for i in h]
    J = (-sum(((Y * np.log(h1))) + ((1 - Y) * np.log(h2))) / m) + ((lambda_reg * sum(theta[1:]**2))/ (2*m))

    grad = (X.transpose().multiply((h-Y))).sum(axis=1).getA()[:, 0] / m
    grad[1:] = grad[1:] + ((lambda_reg * theta[1:]) / m)

    return J, grad

def train(Xtrain, Ytrain, lambda_reg):

    m, n = Xtrain.shape
    X = sparse.csr_matrix(np.column_stack( (np.ones(m),Xtrain.toarray()) ))
    theta = np.zeros(n+1)
    MaxIter = 100

    result = scipy.optimize.minimize(fun=costReg, x0=theta, args=(X, Ytrain, lambda_reg),
                method='L-BFGS-B', jac=True, options={'maxiter': MaxIter, 'disp':True})

    return result.x


def predict(theta, Xtest):

    m = Xtest.shape[0]
    p = np.zeros(m, dtype=int)

    X = sparse.csr_matrix(np.column_stack( (np.ones(m),Xtest.toarray()) ))
    h = sigmoid((X.multiply(theta)).sum(axis=1).getA()[:, 0])
    p = [1 if i >= 0.5 else 0 for i in h]

    return p

def regression(Xtrain, Ytrain, Xtest, lambda_reg):

    theta = train(Xtrain, Ytrain, lambda_reg)

    Ypred = predict(theta, Xtest)

    return theta, Ypred
