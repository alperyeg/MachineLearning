import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class AdalineGD(object):
    """
    Adaptive linear neuron classifier

    """
    def __init__(self, eta=0.01, epochs=10):
        """
        :param eta: float
            Learning rate [0.0, 1.0]
        :param epochs: int
            Number of iterations to pass over trainingset
        :param w_: numpy.array
            shape = [n_features]
            Weights after fitting
        :param errors_: list
            Number of miscalculations in every epoch
        """
        self.eta = eta
        self.epochs = epochs

    def fit(self, X, y):
        """
        Fit training data, training step

        ..math:
            J(w) = \frac{1}{2}\Sigma_i(y_i - phi(z_i))^2
            \phi(z_i) = w^Tx
            w = w + \delta w
            \dela w = \eta * -\nabla J(w)
            \nabla J(w) = - \Sigma_i (y_i - \phi(z_i))x_i

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
        self.cost_ = []

        for i in range(self.epochs):
            # Calculate w^Tx
            output = self.net_input(X)
            errors = (y - output)
            # Gradient for w, \nabla J(w)
            self.w_[0] += self.eta * errors.sum()
            self.w_[1:] += self.eta * np.dot(X.T, errors)
            # Sum squared error (SSE)
            cost = np.square(errors).sum() / 2.0
            self.cost_.append(cost)
        return

    def net_input(self, X):
        """
        Calculate the net input

        ..math:
            w^Tx

        :param X: Feature matrix
        :return: numpy.array
            Dot product of `X` and weights `w_`
        """
        # wTx + w0, w0 is theta and w[1:] are the weights without theta
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X,):
        return np.where(self.activation(X) >= 0.0, 1, -1)


def plot_learning_rate_sse(X, y):
    ada1 = AdalineGD(epochs=10, eta=0.01)
    ada1.fit(X, y)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-error)')
    ax[0].set_title('Adaline - Learning rate 0.01')

    ada2 = AdalineGD(eta=0.0001, epochs=10)
    ada2.fit(X, y)

    ax[1].plot(range(1, len(ada2.cost_) + 1), np.log10(ada2.cost_), marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('log(Sum-squared-error)')
    ax[1].set_title('Adaline - Learning rate 0.0001')
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
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, cmap=cmap(
                idx), marker=markers[idx], label=cl)


def plot_standardized_sse(X, y, eta, epochs):
    X_std = np.copy(X)
    X_std[:, 0] = (X_std[:, 0] - X_std[:, 0].mean()) / X_std[:, 0].std()
    X_std[:, 1] = (X_std[:, 1] - X_std[:, 1].mean()) / X_std[:, 1].std()
    ada = AdalineGD(eta=eta, epochs=epochs)
    ada.fit(X_std, y)
    plot_decision_regions(X_std, y, classifier=ada)
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.title('Adaline eta: {}, epochs: {}'.format(eta, epochs))
    plt.show()

    plt.title('Adaline eta: {}, epochs: {}'.format(eta, epochs))
    plt.xlabel('Epochs')
    plt.ylabel('log(Sum-squared-error)')
    plt.plot(range(1, len(ada.cost_) + 1), np.log10(ada.cost_), marker='o')
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

plot_learning_rate_sse(X, y)
plot_standardized_sse(X, y, eta=0.01, epochs=20)