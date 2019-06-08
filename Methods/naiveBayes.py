import numpy as np
import pandas as pd

def train(Xtrain, Ytrain):

    pAtrTrue = np.zeros(Xtrain.shape[1])
    pAtrFake = np.zeros(Xtrain.shape[1])

    XFake = Xtrain[Ytrain==1]
    XTrue = Xtrain[Ytrain==0]

    pFake = XFake.shape[0]/len(Ytrain)
    pTrue = XTrue.shape[0]/len(Ytrain)

    pAtrFake = (XFake.sum(axis=0).getA()[0] + 1) / (XFake.shape[0] + XFake.shape[1])
    pAtrTrue = (XTrue.sum(axis=0).getA()[0] + 1) / (XTrue.shape[0] + XTrue.shape[1])

    return pAtrFake, pAtrTrue, pFake, pTrue

def predict(x,pFake,pTrue,pAtrFake,pAtrTrue):
    classe = 0;
    probFake= 0;
    probTrue = 0;

    # for i in range(x.shape[0]):
    #     probFake = probFake * (pAtrFake[i] ** x[i])
    #     probTrue = probTrue * (pAtrTrue[i] ** x[i])
    # probFake = pFake * probFake
    # probTrue = pTrue * probTrue

    probTrue = pTrue * (x*pAtrTrue + (1 - x)*(1 - pAtrTrue)).prod()
    probFake = pFake * (x*pAtrFake + (1 - x)*(1 - pAtrFake)).prod()
    classe = 1 if probFake > probTrue else 0

    return classe, probFake, probTrue

def naiveBayes(Xtrain, Ytrain, Xtest):

    pAtrFake, pAtrTrue, pFake, pTrue = train(Xtrain, Ytrain)

    Ypred = np.zeros(0)
    for i in range(Xtest.shape[0]):
        Ypredi, probFake, probTrue = predict(Xtest.getrow(i).toarray(), pFake, pTrue, pAtrFake, pAtrTrue)
        Ypred = np.append(Ypred, Ypredi);

    return Ypred
