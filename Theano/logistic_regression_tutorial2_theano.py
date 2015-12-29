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
    """
    Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """
    def __init__(self, inpt, n_in, n_out):
        """
        :type inpt: theano.tensor.TensorType
        :param inpt: input, one minibatch
        :type n_in: int
        :param n_in: number of input units, dimension of space in which the
            labes lie
        :type n_out: int
        :param n_out: number of output units, dinension of space in the
            labels lie

        """
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
        self.p_y_given_x = T.nnet.softmax(T.dot(inpt, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        # argmax returns the index of maximum value along given axis
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # Parameters of model
        self.params = [self.W, self.b]

        # save input
        self.inpt = inpt

    def negative_likelihood(self, y):
        """
        Return the mean of the negative log-likelihood of the prediction of this
        model under a given target distribution.
        .. math::
        \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
        \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
        \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
        \ell (\theta=\{W,b\}, \mathcal{D})

         :type y: theano.tensor.TensorType
         :param y: corresponds to a vector that gives for each example the
              correct label
         Note: mean instead of the sum is used so that the learning rate is less
         dependent on the batch size
         """

        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class. LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x))

    def errors(self, y):
        """
        Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one loss over
         the size of the minibatch
        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
              correct label
        """
        # check if same dimension
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred))
        # check for correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
