from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# utility generic functions that will be used
# by both neural networks


def init_weight_and_bias(M1, M2):
  # initialize the weight matrix and the bias
  # M1 is input size, M2 is the outuput size
  
  # create a matrix of M1 x M2, which is randomized initially to a Gaussian normal (W expression numerator)
  # make a standard deviation of the square root of fan-in (W expression denominator)
  W = np.random.randn(M1, M2) / np.sqrt(M1)

  # the bias is initialized in zeros, and that's just size M2
  b = np.zeros(M2)

  # turn these into float32 to use them in Theano and Tensorflow without problems
  return W.astype(np.float32), b.astype(np.float32)


def init_filter(shape, poolsz):
  # used by convolutional neural networks
  # shape will be a tuple of four different values 

  # divide by fan-in plus fan-out
  w = np.random.randn(*shape) * np.sqrt(2) / np.sqrt(
    np.prod(shape[1:]) + shape[0]*np.prod(shape[2:] / np.prod(poolsz)))

  # convert to float32 for the same reasons above
  return w.astype(np.float32)


def relu(x):
  # rectifier linear unit function (relu)
  # used for activation function inside the neural network
  # can be used with older versions of Theano that does not
  # have relu built in
  return x * (x > 0)


def sigmoid(A):
  return 1 / (1 + np.exp(-A))


def softmax(A):
  expA = np.exp(A)
  return expA / expA.sum(axis=1, keepdims=True)


def sigmoid_cost(T, Y):
  # calculates the cross entropy from the definition
  # for sigmoid cost (for binary classification)
  return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()


def cost(T, Y):
  # more general cross entropy function
  # work for softmax (whereas this function is direct from the definition) 
  return -(T*np.log(Y)).sum()


def cost2(T, Y):
  # same as cost(), just uses the targets to index Y that the values are non-zero
  N = len(T)
  return -np.log(Y[np.arange(N), T]).mean()


def error_rate(targets, predictions):
  # returns the error rate between targets and predictions
  return np.mean(targets != predictions)


def y2indicator(y):
  # turns a N x 1 vector of targets
  # so that will have the class labels 0..K-1
  # that turns it into an indicator matrix
  # which'll only have the values 0 and 1
  # but the size'll be N x K

  N = len(y)
  K = len(set(y))
  ind = np.zeros((N, K))
  for i in range(N):
      ind[i, y[i]] = 1
  return ind


def getData(balance_ones = True, Ntest = 1000):
  # get all the data from all the classes

  # images are 48x48 = 2304 size vectors
  Y = []
  X = []
  first = True # skipping headers

  for line in open('data/fer2013.csv'):
    if first:
      first = False
    else:
      row = line.split(',')
      Y.append(int(row[0])) # first column is label
      X.append([int(p) for p in row[1].split()]) # second column are space separeted pixels
  
  # convert these into NumPy arrays
  # and normalize this data (so X goes 0..1 instead of 0..255)
  X, Y = np.array(X) / 255.0, np.array(Y)

  # shufle and split
  X, Y = shuffle(X,Y)
  Xtrain, Ytrain = X[:-Ntest], Y[:-Ntest]
  Xvalid, Yvalid = X[-Ntest:], Y[-Ntest:]

  if balance_ones:
    # knowing that the classes are imbalanced
    # balance the 1 class

    # take all the data that is not of class 1
    X0, Y0 = Xtrain[Ytrain != 1, :], Ytrain[Ytrain != 1]

    # get samples where Y == 1
    X1 = Xtrain[Ytrain == 1, :]
    
    # repeat it 9 times
    X1 = np.repeat(X1, 9, axis = 0)
    
    # stacks X0 and X1 back together
    Xtrain = np.vstack([X0, X1])

    # same for Y0 and Y1
    Ytrain = np.concatenate((Y0, [1]*len(X1)))

  return Xtrain, Ytrain, Xvalid, Yvalid


def getImageData():
  # used by convolutional neural networks
  # keeps the original image shape

  X, Y, _, _ = getData()
  N, D = X.shape
  d = int(np.sqrt(D))

  # N -> number of samples
  # 1 -> color channels
  # first d -> weight
  # last d -> height
  X = X.reshape(N, 1, d, d)

  # Xtrain, Ytrain, Xvalid, Yvalid = getData()
  # N, D = Xtrain.shape

  # Xtrain = Xtrain.reshape(-1, 1, d, d)
  # Xvalid = Xvalid.reshape(-1, 1, d, d)
  # return Xtrain, Ytrain, Xvalid, Yvalid
  return X, Y


def getBinaryData():
  # same of getData()
  # the difference is that this only add
  # the samples for which the classes is 0 or 1

  Y = []
  X = []
  first = True
  for line in open('fer2013.csv'):
    if first:
      first = False
    else:
      row = line.split(',')
      y = int(row[0])
      if y == 0 or y == 1:
        Y.append(y)
        X.append([int(p) for p in row[1].split()])
  return np.array(X) / 255.0, np.array(Y)


def crossValidation(model, X, Y, K=5):
  # split data into K parts
  X, Y = shuffle(X, Y)
  sz = len(Y) // K
  errors = []
  for k in range(K):
    xtr = np.concatenate([X[:k*sz, :], X[(k*sz + sz):, :]])
    ytr = np.concatenate([Y[:k*sz], Y[(k*sz + sz):]])
    xte = X[k*sz:(k*sz + sz), :]
    yte = Y[k*sz:(k*sz + sz)]

    model.fit(xtr, ytr)
    err = model.score(xte, yte)
    errors.append(err)
  print("errors:", errors)
  return np.mean(errors)
