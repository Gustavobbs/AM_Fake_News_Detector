import numpy as np
import pandas as pd
import scipy
import scipy.optimize
from scipy import sparse

def sigmoid(z):

    return z = 1/(1+np.exp(-z))

def sigmoidGradient(z):

    return sigmoid(z) * (1 - sigmoid(z))

def funcaoCusto(nn_params, Xtrain, Ytrain, vLambda):
    m, input_layer_size = Xtrain.shape
    hidden_layer_size = 500

    Theta1 = np.reshape( nn_params[0:hidden_layer_size*(input_layer_size + 1)], (hidden_layer_size, input_layer_size+1) )
    Theta2 = np.reshape( nn_params[ hidden_layer_size*(input_layer_size + 1):], (1, hidden_layer_size+1) )

    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    X = sparse.csr_matrix(np.column_stack( (np.ones(m),Xtrain.toarray()) ))
    z2 = Theta1.dot(X.transpose())
    a2 = sigmoid(z2)
    a2 = sparse.csr_matrix(np.column_stack( (np.ones(m),a2.transpose().toarray()) ))
    z3 = Theta2.dot(a2.transpose())
    a3 = sigmoid(z3).toarray()

    J = (-sum(((Y * np.log(a3))) + ((1 - Y) * np.log(1-a3))) / m)
    reg = (Theta1[:, 1:] ** 2).sum() + (Theta2[:, 1:] ** 2).sum()
    reg = reg * (vLambda / (2 * m))
    J = J + reg

    delta3 = a3 - Y
    delta2 = Theta2.transpose().dot(delta3)[1:,].multiply(sigmoidGradient(z2))
    Theta2_grad = Theta2_grad + delta3.trannspose().dot(a2)
    Theta1_grad = Theta1_grad + delta2.transpose().dot(X)
    Theta2_grad = (Theta2_grad / m) + (Theta2[:, 1:] * (vLambda / m))
    Theta1_grad = (Theta1_grad / m) + (Theta1[:, 1:] * (vLambda / m))

    grad = np.concatenate([np.ravel(Theta1_grad), np.ravel(Theta2_grad)])

    return J, grad

def train(Xtrain, Ytrain, vLambda):

    m, n = X.train.shape
    MaxIter = 500
    epsilon_init = 0.12
    initialTheta1 = np.random.RandomState(10).rand(500, 1 + n) * 2 * epsilon_init - epsilon_init
    initialTheta2 = np.random.RandomState(20).rand(1, 501) * 2 * epsilon_init - epsilon_init
    initial_rna_params = np.concatenate([np.ravel(initial_Theta1), np.ravel(initial_Theta2)])

    result = scipy.optimize.minimize(fun=funcaoCusto, x0=initial_rna_params, args=(Xtrain, Ytrain, vLambda),
                    method='TNC', jac=True, options={'maxiter': MaxIter})

    nn_params = result.x

    # Obtem Theta1 e Theta2 back a partir de rna_params
    Theta1 = np.reshape( nn_params[0:500*(n + 1)], (500, n+1) )
    Theta2 = np.reshape( nn_params[ 500*(n + 1):], (1, 501) )

    return Theta1, Theta2

def predict(Theta1, Theta2, Xtest):

    m = X.shape[0]
    num_labels = Theta2.shape[0]

    a1 = sparse.csr_matrix(np.column_stack( (np.ones(m),Xtest.toarray()) ))
    h1 = sigmoid( a1.dot(Theta1.transpose()) )

    a2 = sparse.csr_matrix(np.column_stack( (np.ones(m),h1.toarray()) ))
    h2 = sigmoid( a2.dot(Theta2.transpose()) )

    p = [1 if i >= 0.5 else 0 for i in h2]

    return p

def neuralNetwork(Xtrain, Ytrain, Xtest, vLambda):

    Theta1, Theta2 = train(Xtrain, Ytrain, vLambda)

    Ypred = predict(Theta1, Theta2, Xtest)

    return Ypred, Theta1, Theta2
