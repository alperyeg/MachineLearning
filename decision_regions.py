from itertools import cycle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_decision_regions(X, y, clf, X_highlight=None, res=0.02, cycle_marker=True, legend=1, cmap=None):
    """
    Plots decision regions of a classifier.
    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
      Feature Matrix.
    y : array-like, shape = [n_samples]
      True class labels.
    clf : Classifier object. Must have a .predict method.
    X_highlight : array-like, shape = [n_samples, n_features] (default: None)
      An array with data points that are used to highlight samples in `X`.
    res : float (default: 0.02)
      Grid width. Lower values increase the resolution but
      slow down the plotting.
    cycle_marker : bool
      Use different marker for each class.
    legend : int
      Integer to specify the legend location.
      No legend if legend is 0.
    cmap : Custom colormap object .
    Returns
    ---------
    None
    Examples
    --------
    from sklearn import datasets
    from sklearn.svm import SVC
    iris = datasets.load_iris()
    X = iris.data[:, [0,2]]
    y = iris.target
    svm = SVC(C=1.0, kernel='linear')
    svm.fit(X,y)
    plot_decision_region(X, y, clf=svm, res=0.02, cycle_marker=True, legend=1)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.title('SVM on Iris')
    plt.show()
    """
    # check if data is numpy array
    for a in (X, y):
        if not isinstance(a, np.ndarray):
            raise ValueError('%s must be a NumPy array.' % a.__name__)

    # check if test data is provided
    plot_testdata = True
    if not isinstance(X_highlight, np.ndarray):
        if X_highlight is not None:
            raise ValueError('X_test must be a NumPy array or None')
        else:
            plot_testdata = False

    if len(X.shape) == 2 and X.shape[1] > 1:
        dim = '2d'
    else:
        dim = '1d'


    marker_gen = cycle('sxo^v')

    # make color map
    colors = ['red', 'blue', 'lightgreen', 'gray', 'cyan']
    n_classes = len(np.unique(y))
    if n_classes > 5:
        raise NotImplementedError('Does not support more than 5 classes.')

    if not cmap:
        cmap = matplotlib.colors.ListedColormap(colors[:n_classes])

    # plot the decision surface

    if dim == '2d':
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    else:
        y_min, y_max = -1, 1

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, res),
                         np.arange(y_min, y_max, res))

    if dim == '2d':
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        Z = clf.predict(np.array([xx.ravel(), yy.ravel()]).T)
    else:
        y_min, y_max = -1, 1
        Z = clf.predict(np.array([xx.ravel()]).T)

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    # plot class samples

    for c in np.unique(y):
        if dim == '2d':
            y_data = X[y==c, 1]
        else:
            y_data = [0 for i in X[y==c]]

        plt.scatter(x=X[y==c, 0],
                    y=y_data,
                    alpha=0.8,
                    c=cmap(c),
                    marker=next(marker_gen),
                    label=c)

    if legend:
        plt.legend(loc=legend, fancybox=True, framealpha=0.5)

    if plot_testdata:
        if dim == '2d':
            plt.scatter(X_highlight[:,0], X_highlight[:,1], c='', alpha=1.0, linewidth=1, marker='o', s=80)
        else:
            plt.scatter(X_highlight, [0 for i in X_highlight], c='', alpha=1.0, linewidth=1, marker='o', s=80)

def plot_decision_regions_(X, y, classifier, test_idx=None,
                          resolution=0.002):
    from matplotlib.colors import ListedColormap
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='',
                    alpha=1.0, linewidth=1, marker='o',
                    s=55, label='test set')
