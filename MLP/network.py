from __future__ import division
import random
import numpy as np


class Network(object):
    def __init__(self, sizes):
        # Number of neurons, e.g. net = Network([2,3,1]) is a network with 2
        # neurons in first layer, 3 neurons in second layer, 1 neuron in
        # third layer etc.
        self.sizes = sizes
        # Set number of layers
        self.num_layer = len(sizes)
        # Random biases and weights, Gaussian distribution [0,1]
        # Create bias matrix
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # Create weight matrix w_jk, kth neuron from 2nd layer to jth neuron
        #  in 3rd layer
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feed_forward(self, a):
        """
        Return output of Network
        Calculates activation function:
            ..math:: \sigma(wa + b)
        
        """
        for b, w in zip(self.biases, self.weights):
            # TODO: Make this without vectorized function
            a = sigmoid_vec(np.dot(w, a) + b)
            return a

    def stochastic_gradient_descent(self, training_data, epochs, batch_size,
                                    eta, test_data=None):
        """
        Training the network using mini-batch stochastic gradient descent.
        The training data is a list of tuples `(x, y)`, where `x` is the input
        and `y` the desired output.
        If :attr:`test_data`is given then the network will be evaluated
        against the test data after each epoch, but will slow down the whole
        process.

        :param training_data: List of tuples, with input and target output
        :param int epochs: Number of epochs to train used in mini-batch
        :param int batch_size: Size of the batch used in mini-batch
        :param float eta: learning rate
        :param test_data: Data used for cross testing (optional)
        """

        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            # Shuffle the input data
            random.shuffle(training_data)
            # Create mini batches out the training data with given size size
            mini_batches = [training_data[k:k + batch_size] for k in
                            xrange(0, n, batch_size)]
            # Update all batches
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(
                    test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, batch, eta):
        """
        Update the network's weights and biases by applying gradient descent
        using backpropagation to given mini-batch.

        :param batch: Mini-batch which will be updated, List of tuples (x,y)
        :param int eta: Learning rate

        """
        nabla_bias = [np.zeros(b.shape) for b in self.biases]
        nabla_weights = [np.zeros(w.shape) for w in self.weights]
        # Calculate gradient for weights and biases
        for x, y in batch:
            # Calculate now the partial derivatives using backpropagation
            delta_nabla_weights, delta_nabla_biases = self.back_propagation(
                x, y)
            nabla_bias = [nb + dnb for nb, dnb in zip(nabla_bias,
                                                      delta_nabla_biases)]
            nabla_weights = [nw + dnw for nw, dnw in zip(nabla_weights,
                                                         delta_nabla_weights)]
        # Update weights and biases described by the gradient method
        self.weights = [w - (eta / len(batch)) * nw for w, nw in zip(
            self.weights, nabla_weights)]
        self.biases = [b - (eta / len(batch) * nb for b, nb in zip(
            self.biases, nabla_bias))]

    def evaluate(self, test_data):
        """
        Number of test inputs, which are correctly classified.
        The neural network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.
        :param test_data: test data
        :return: number of correctly classified inputs (int)
        """
        test_results = [np.argmax(self.feed_forward(x), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def back_propagation(self, x, y):
        """
        Returns tuple (nabla_b, nabla_w) as gradient of cost function `C` (
        `C_x`). `nabla_w` and `nabla_b` are calculated layer-wise.
        """
        nabla_bias = [np.zeros(b.shape) for b in self.biases]
        nabla_weights = [np.zeros(w.shape) for w in self.weights]
        # Calculate feed-forward
        activation = x
        # Create list to store activations layer by layer
        activations = [x]
        # List to store vectors layer by layer
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid_vec(activation)
            activations.append(activation)
        # backward propagation steps
        # output error: delta vector
        delta = self.cost_derivatives(activations[-1], y) * sigmoid_prime(
            zs[-1])
        # TODO probably unnecessary
        nabla_bias[-1] = delta
        nabla_weights[-1] = np.dot(delta, activations[-2].transpose())
        # back-propagate errors
        for l in xrange(2, self.num_layer):
            z = zs[-l]
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * \
                    sigmoid_prime_vec(z)
            nabla_bias[-l] = delta
            nabla_weights[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_bias, nabla_weights


    @staticmethod
    def cost_derivatives(output_activations, target):
        return output_activations - target


def sigmoid(z):
    """
    Calculates sigmoid function
    :param z: input
    :return: result of sigmoid function
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """
    Calculates the partial derivative of sigmoid
    :param z: input
    :return: result of derivative of sigmoid function

    """
    sigma = sigmoid(z)
    return sigma * (1 - sigma)

# Define vectorized function, which returns a numpy array
sigmoid_vec = np.vectorize(sigmoid)
sigmoid_prime_vec = np.vectorize(sigmoid_prime)


# TODO
def vectorized(x):
    pass