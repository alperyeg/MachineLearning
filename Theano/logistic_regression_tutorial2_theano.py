"""
Logistic regression is a probabilistic, linear classifier. It is
parametrized by a weight matrix W and a bias vector b. Classification is done
by projecting an input vector onto a set of hyperplanes, each of which
corresponds to a class. The distance from the input to a hyperplane reflects
the probability that the input is a member of the corresponding class.

Mathematically, the probability that an input vector x is a member of a
class i, a value of a stochastic variable Y , can be written as:

.. math :: P(Y = i | x, W, b) = softmax_i(Wx + b) = \frac{\exp(W_ix + b_i)}{
\sum_j \exp(W_jx + b_j)}

The model's prediction y_pred is the class whose probability is maximal,
specifically:
.. math :: y_pred = argmax_i(P (Y = i| x, W, b))
"""
import theano
import theano.tensor as T
import numpy as np

class LogisticRegression(object):
    def __init__(self, n_in, n_out):
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=np.zeros((n_in, n_out),
                                              dtype=theano.config.floatX),
                               name='W', borrow=True)
        # init bias b as a vector of n_out O's
        self.b = theano.shared(value=np.zeros(n_out,
                                              dtype=theano.config.floatX),
                               name='b', borrow=True)

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        #  probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)



