from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt

# import getData function from util file
from util import getData

# label instance
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def main():
  # get images
  X, Y, _, _ = getData(balance_ones=False)

  # (almost) infinity loop
  while True:
    # looping between the emotions
    for i in xrange(7):
      # get image related with these emotions
      x, y = X[Y==i], Y[Y==i]

      # get the number of datapoints that belongs to current emotion
      N = len(y)

      # choose one datapoint randomly
      j = np.random.choice(N)

      # and plot it
      plt.imshow(x[j].reshape(48, 48), cmap = 'gray')
      plt.title(label_map[y[j]])
      plt.show()

    # dialog box
    prompt = raw_input('Quit? Enter Y: \n')
    if prompt == 'Y':
      break


if __name__ == '__main__':
  main()
