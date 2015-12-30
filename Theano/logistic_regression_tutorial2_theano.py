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
import gzip
import os

import cPickle
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


def load_data(dataset):
    """
    Loads the dataset
    :type dataset: string
    :param dataset: path to dataset, mnist dataset
    :return:
    """
    dataset_name = 'mnist.pkl.gz'
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        new_path = os.path.join(
            os.path.split(__file__)[0], "..", "data", dataset)
        if os.path.isfile(new_path) or data_file == dataset_name:
            dataset = new_path
    # if dataset not found download it
    if (not os.path.isfile(dataset)) and data_file == dataset_name:
        import urllib
        origin = ('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz')
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)
    # load dataset
    f = gzip.open(dataset, 'rb')
    train_set, validate_set, test_set = cPickle.load(f)
    f.close()
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is an numpy.ndarray of 2 dimensions (a matrix)
    # which row's correspond to an example. target is a
    # numpy.ndarray of 1 dimensions (vector)) that have the same length as
    # the number of rows in the input. It should give the target to the example
    # with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """
        Function that loads the dataset into theano shared variables

        Easier to load into GPU and to work with it.
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a
        shared variable) would lead to a large decrease in performance.

        :param data_xy: whole mnist train, test or validate data set
        :param borrow: invokes theanos parameter borrow
        :return: list of tuples (data, label) of theano arrays with train,
        test and validate data sets
        """
        # split tuple into data (data_x) and label (data_y)
        data_x, data_y = data_xy
        # now create theano arrays
        shared_x = theano.shared(np.asarray(data_x, dtype=T.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=T.config.floatX),
                                 borrow=borrow)
        # When storing to GPU it should be floats.
        # But during computation the labes are needed as ints.
        # That's why they are casted.
        return shared_x, T.cast(shared_y, dtype='int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(validate_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
                           dataset='mnist.pkl.gz', batch_size=600):
    """
    Stochastic gradient descent  of log-linear model

    :type learning_rate: float
    :param learning_rate: learning rate for the sgd
    :type n_epochs: int
    :param n_epochs: number of epochs used
    :type dataset: string
    :param dataset: path of dataset
    :type batch_size: int
    :param batch_size: number of batches to be used
    :return:

    """
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    print '... building model'

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a minibatch)
    # data, presented as rasterized images
    x = T.matrix(name='x')
    # labels, presented as 1D vector of [int] labels
    y = T.matrix(name='y')

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(inpt=x, n_in=28 * 28, n_out=10)
    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(inputs=[index],
                                 outputs=classifier.errors(y),
                                 givens={x: test_set_x[
                                            index * batch_size: (index + 1) *
                                                                batch_size],
                                         y: test_set_y[
                                            index * batch_size: (index + 1) *
                                                                batch_size]})
    validate_model = theano.function(inputs=[index],
                                     outputs=classifier.errors(y),
                                     givens={
                                         x: valid_set_x[
                                            index * batch_size: (index + 1) *
                                                                batch_size],
                                         y: valid_set_y[
                                            index * batch_size: (index + 1) *
                                                                batch_size]})
    # calculate gradient cost wrt to W and b (= theta)
    g_W = theano.grad(cost=cost, wrt=classifier.W)
    g_b = theano.grad(cost=cost, wrt=classifier.b)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    train_model = theano.function(inputs=[index],
                                  outputs=cost,
                                  updates=updates,
                                     givens={
                                         x: train_set_x[
                                            index * batch_size: (index + 1) *
                                                                batch_size],
                                         y: train_set_y[
                                            index * batch_size: (index + 1) *
                                                                batch_size]})

    print '... training the model'
