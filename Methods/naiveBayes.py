import numpy as np
import pandas as pd

def train(Xtrain, Ytrain):

    pAtrTrue = np.zeros(Xtrain.shape[1])
    pAtrFake = np.zeros(Xtrain.shape[1])

    XFake = Xtrain[Ytrain==1]
    XTrue = Xtrain[Ytrain==0]

    pFake = XFake.shape[0]/len(Ytrain)
    pTrue = XTrue.shape[0]/len(Ytrain)

    pAtrFake = (XFake.sum(axis=0) + 1) / (XFake.shape[0] + XFake.shape[1])
    pAtrTrue = (XTrue.sum(axis=0) + 1) / (XTrue.shape[0] + XFake.shape[1])

    return pAtrFake, pAtrTrue, pFake, pTrue

def predict(x,pFake,pTrue,pAtrFake,pAtrTrue):
    classe = 0;
    probFake= 1;
    probTrue = 1;

    display(x.shape)
    for i in range(x.shape[0]):
        probFake = probFake * (pAtrFake[i] ** x[i])
        probTrue = probTrue * (pAtrTrue[i] ** x[i])
    probFake = pFake * probFake
    probTrue = pTrue * probTrue
    classe = 1 if probFake > probTrue else 0

    return classe, probFake, probTrue

def naiveBayes(Xtrain, Ytrain, Xtest):

    display(Xtest.shape)
    pAtrFake, pAtrTrue, pFake, pTrue = train(Xtrain, Ytrain)

    display(pAtrTrue, pAtrFake, pFake, pTrue)

    Ypred = [predict(i, pFake, pTrue, pAtrFake, pAtrTrue) for i in Xtest]

    return Ypred
