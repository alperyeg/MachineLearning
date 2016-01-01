import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# import seaborn
# seaborn.color_palette(palette='pastel')

class Perceptron(object):
    """
    Perceptron classifier

    :param w_: numpy.array
        shape = [n_features]
        Weights after fitting
        :param errors_: list
        Number of miscalculations in every epoch
    """

    def __init__(self, eta=0.01, epochs=10):
        """

        :param eta: float
            Learning rate [0.0, 1.0]
            :param epochs: int
            Number of iterations to pass over trainingset
        """
        self.eta = eta
        self.epochs = epochs
        self.errors_ = []

    def fit(self, X, Y):
        """
        Fit training data, training step

        :param X: numpy.array
            shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features
        :param Y: numpy.array
            shape=[n_samples]
            Target values
        :return: self: object
        """
        # Weight is a vector of size of the features + 1 which is the
        # threshold theta
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, Y):
                # delta_w = eta(target_i-output_i)x_i
                delta_w = self.eta * (target - self.predict(xi))
                # w = w + delta_w
                self.w_[1:] += delta_w * xi
                self.w_[0] += delta_w
                errors += int(delta_w != 0.)
            # error per epoch
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """
        Calculate the net input

        ..math:
            w^Tx

        :param X:
        :return:
        """
        # wTx + w0, w0 is theta and w[1:] are the weights without theta
        return np.dot(X, self.w_[1:] + self.w_[0])

    def predict(self, X):
        """
        Return class label after unit step
        :param X:
        :return:
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)


def scatterplot(X, y):
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:, 0], X[50:, 1], color='blue', marker='x',
                label='versicolor')
    plt.xlabel('petal length')
    plt.ylabel('sepal length')
    plt.legend(loc='upper left')
    plt.show()


def plot_miscalculations(classifier, X, y):
    print('Weights: {}'.format(classifier.w_))
    print('Errors: {}'.format(classifier.errors_))
    plt.plot(range(1, len(classifier.errors_) + 1), classifier.errors_,
             marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of miscalculations')
    plt.show()


def plot_decision_regions(X, y, classifier, resolution=0.22):
    from matplotlib.colors import ListedColormap
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightblue', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot decision surface
    x1_min, x1_max = np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1
    x2_min, x2_max = np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1
    # create a pair of grid arrays
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(
            x2_min, x2_max, resolution))
    # classifier model is trained on two feature dimensions, we need to
    # flatten the grid arrays and create a matrix which has the same number
    # of columns as the Iris data subset, then it is possible to use the
    # predict method to predict the class labels
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # reshape predicted labels into grid with same dimension as xx1 and xx2
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    #plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, cmap=cmap(
                idx), marker=markers[idx], label=cl)
    plt.xlabel('sepal length in cm')
    plt.ylabel('petal length in cm')
    plt.legend(loc='upper left')
    plt.show()


# load iris data set
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases'
                 '/iris/iris.data', header=None)

# choose setose and versicolor, from 0 to 100
y = df.iloc[0:100, 4].values
# transform to labels +1 and -1, where setosa is -1 and versicolor is +1
y = np.where(y == 'Iris-setosa', -1, 1)

# sepal length and petal length, column 0 and 2
X = df.iloc[:100, [0, 2]].values
scatterplot(X,y)
ppn = Perceptron(epochs=10, eta=0.1)
ppn.fit(X, y)
plot_miscalculations(ppn, X, y)
plot_decision_regions(X, y, ppn)
