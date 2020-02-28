import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from util import getBinaryData, sigmoid, sigmoid_cost, error_rate, relu

class ANN(object):
  # pass the number of hidden units
  def __init__(self, M):
    self.M = M

  def fit(self, X, Y, learning_rate = 5 * 10e-7, reg = 1.0, epochs = 10000, show_fig = False):
    X, Y = shuffle(X, Y)

    Xvalid, Yvalid = X[-1000:], Y[-1000:] # the last 1k datapoints will be the validation
    X, Y = X[:-1000], Y[:-1000] # remaining datapoints
    N, D = X.shape # get the shape of X
    

    # starting weights

    self.W1 = np.random.randn(D, self.M) / np.sqrt(D + self.M) # first weight
    self.b1 = np.zeros(self.M) # bias

    self.W2 = np.random.rand(self.M) / np.sqrt(self.M) # hidden output weights
    self.b2 = 0


    costs = []
    best_validation_error = 1

    for i in xrange(epochs):
      # forward propagation
      
      # pY -> probability of Y
      # Z -> hidden layer value
      pY, Z = self.forward(X)

      # gradient descent step

      pY_Y = pY - Y # prediction - target

      # reg -> regularization
      self.W2 -= learning_rate * (Z.T.dot(pY_Y) + reg * self.W2) # update outputs to hidden weights
      self.b2 -= learning_rate * ((pY_Y).sum() + reg * self.b2) # update bias


      # back propagation on the input to hidden weights
      
      dZ = np.outer(pY_Y, self.W2) * (Z > 0)
      self.W1 -= learning_rate * (X.T.dot(dZ) + reg * self.W1) # update input to hidden weights
      self.b1 -= learning_rate * (np.sum(dZ, axis = 0) + reg * self.b1) # update bias


      # print out the validation error and the validation cost by 20 times
      
      if i % 20 == 0:
        pYvalid, _ = self.forward(Xvalid)

        c = sigmoid_cost(Yvalid, pYvalid) # calculates validation costs (using cross entropy)
        e = error_rate(Yvalid, np.round(pYvalid)) # calculates validation error
        print "i:", i, "cost:", c, "error:", e

        if e < best_validation_error: # if the error rate is less then the current best validation error rate
          best_validation_error = e # this is the new best validation error rate! o/
    
    print "best_validation_error:", best_validation_error

    if show_fig:
      plt.plot(costs)
      plt.show()

    
  # main functions

  def forward(self, X):
    # activation function for the neural network
    Z = relu(X.dot(self.W1) + self.b1)
    return sigmoid(Z.dot(self.W2) + self.b2), Z
  

  def predict(self, X):
    pY, _ = self.forward(X)
    return np.round(pY)


  def score(self, X, Y):
    prediction = self.predict(X)
    return 1 - error_rate(Y, prediction)


def main():
  # get data
  X, Y = getBinaryData()

  # balancing classes samples
  
  X0 = X[Y == 0, :]
  X1 = X[Y == 1, :]
  X1 = np.repeat(X1, 9, axis = 0) # repeat X1 data to make the number of class 0 and class 1 samples equal
  X = np.vstack([X0, X1]) # put all together again
  Y = np.array([0] * len(X0) + [1] * len(X1))


  # create model instance
  model = ANN(100) # set the hidden layer size to 100

  # train the model
  model.fit(X, Y, show_fig=True)

if __name__ == '__main__':
  main()
