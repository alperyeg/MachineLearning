import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class AdalineSGD(object):
    """
    Adaptive Linear Neuron Stochastic Gradient Descent Classifier
    """
    def __init__(self, eta=0.01, epochs=10, shuffle=True, random_state=None):
        """
        :param eta: float
            Learning rate [0.0, 1.0]
        :param epochs: int
            Number of iterations to pass over trainingset
        :param w_: numpy.array
            shape = [n_features]
            Weights after fitting
        :param shuffle : bool (default True)
            If True, shuffles training data every epoch to prevent cycles
        :param random_state: int (default int)
            Set random state for shuffling and initializing the weights
        """
        self.eta = eta
        self.epochs = epochs
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            np.random.seed(random_state)
        self.w_ = None
        self.cost_ = []

    def fit(self, X, y):
        """
        :param X: numpy.array
            shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features
        :param Y: numpy.array
            shape=[n_samples]
            Target values
        :return: self: object
        """
        self._initialize_weights(X.shape[1])

        for i in range(self.epochs):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)

    def partial_fit(self, X, y):
        """
        Fit training data without reinitializing the weights
        """
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _initialize_weights(self, m):
        """
        Init weights to zero
        """
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    @staticmethod
    def _shuffle(X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _update_weights(self, xi, target):
        output = self.net_input(xi)
        errors = (target - output)
        # Gradient for w, \nabla J(w)
        self.w_[0] += self.eta * errors
        self.w_[1:] += self.eta * np.dot(xi.T, errors)
        cost = 0.5 * errors**2
        return cost

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X,):
        return np.where(self.activation(X) >= 0.0, 1, -1)

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


# load iris data set
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases'
                 '/iris/iris.data', header=None)

# choose setose and versicolor, from 0 to 100
y = df.iloc[0:100, 4].values
# transform to labels +1 and -1, where setosa is -1 and versicolor is +1
y = np.where(y == 'Iris-setosa', -1, 1)

# sepal length and petal length, column 0 and 2
X = df.iloc[:100, [0, 2]].values
X_std = np.copy(X)
X_std[:, 0] = (X_std[:, 0] - X_std[:, 0].mean()) / X_std[:, 0].std()
X_std[:, 1] = (X_std[:, 1] - X_std[:, 1].mean()) / X_std[:, 1].std()

ada = AdalineSGD(epochs=15, eta=0.01, random_state=1)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, ada)
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.title('Adaline Stochastic Gradient Descent')
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()