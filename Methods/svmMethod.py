import numpy as np
import pandas as pd
import svmutil
from svmutil import svm_read_problem
from svmutil import svm_problem
from svmutil import svm_parameter
from svmutil import svm_train
from svmutil import svm_predict
from svmutil import svm_save_model

def train(Xtrain, Ytrain, kernel, cost, gamma):

    if kernel == 0:
        problem = svm_problem(Ytrain, Xtrain)
        param = svm_parameter('-c %f -t %d -b 1' %(cost, kernel))
        model = svm_train(problem, param)
    else:
        model = svm_train(Ytrain, Xtrain, '-c %f -t %d -g %f' %(cost, kernel, gamma))

    return model

def predict(model, Xtest, Ytest):

    Ypred, acc, val = svm_predict(Ytest, Xtest, model, "-q")

    return Ypred

def svmUse(Xtrain, Ytrain, Xtest, Ytest, kernel, cost, gamma):

    model = train(Xtrain, Ytrain, kernel, cost, gamma)

    Ypred = predict(model, Xtest, Ytest)

    return model, Ypred
