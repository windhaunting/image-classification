import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
   # compute the loss and the gradient
  num_classes = W.shape[1]          #D
  num_train = X.shape[0]            # N
  loss = 0.0
  for i in xrange(num_train):
      scores = X[i].dot(W)                    # C
      #if j == y[i]:
      #  continue
      
      # get softmax denominator
      scores -= np.max(scores)     #shift to make numeric stablity
      normSoftMaxDenominator = np.exp(scores)/np.sum(np.exp(scores))         #normalized denominator
      lognormSoftMax = np.log(np.sum(normSoftMaxDenominator))-np.log(normSoftMaxDenominator[y[i]])    # -loga/b = logb -log a
      loss += lognormSoftMax
      
      dW[:, y[i]] -= X[i]                 #j = y[i]
      for j in xrange(num_classes): 
          dW[:, j] += np.exp(scores[j]) / np.sum(np.exp(scores)) * X[i]          # derivative dl/dw
          
  #average the loss and dW
  loss /= num_train
  dW /= num_train           # average dW
  
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W   # regularization  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]            # N

  scores = X.dot(W)         #  dimension N X C 

  scores -= np.max(scores, axis = 1)[:, np.newaxis]     # along the C 
  softDenominator = np.exp(scores)
  sumSoftDenominator = np.sum(softDenominator, axis = 1)          # (N, )
  
  correct_class_score = scores[range(num_train), y]

  loss = np.sum(np.log(sumSoftDenominator)) - np.sum(correct_class_score)


  
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  
  
  dsoftMax = np.exp(scores) / sumSoftDenominator.reshape(num_train, 1)      # no -log
  dsoftMax[range(num_train), y] -= 1
  dW = np.dot(X.T, dsoftMax)

  dW = dW / num_train + reg * W
  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

