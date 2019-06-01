import numpy as np
import pandas as pd

def train(Xtrain, Ytrain):

    pAtrTrue = np.zeros(X.shape[1])
    pAtrFake = np.zeros(X.shape[1])

    XFake = X[Y==1]
    XTrue = X[Y==0]

    pFake = XFake.shape[0]/len(Y)
    pTrue = XTrue.shape[0]/len(Y)

    pAtrFake = (XFake.sum(axis=0) + 1) / (XFake.shape[0] + XFake.shape[1])
    pAtrTrue = (XTrue.sum(axis=0) + 1) / (XTrue.shape[0] + XFake.shape[1])

    return pAtrFake, pAtrTrue, pFake, pTrue

def predict(x,pFake,pTrue,pAtrFake,pAtrTrue):
    classe = 0;
    probFake= 1;
    probTrue = 1;


    for i in range(x.shape[1]):
        probFake = probFake * (pAtrFake[i] ** x[i])
        probTrue = probTrue * (pAtrTrue[i] ** x[i])
    probFake = pFake * probFake
    probTrue = pTrue * probTrue
    classe = 1 if probFake > probTrue else 0

    return classe, probVitoria, probDerrota

def naiveBayes(Xtrain, Ytrain, Xtest):

    pAtrFake, pAtrTrue, pFake, pTrue = train(Xtrain, Ytrain)

    Ypred = [predict(i, pFake, pTrue, pAtrFake, pAtrTrue) for i in Xtest]

    return Ypred
