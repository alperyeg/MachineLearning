from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from decision_regions import plot_decision_regions_
import matplotlib.pyplot as plt
import numpy as np

# load iris data-set
iris = datasets.load_iris()
# last two features of iris dataset
X = iris.data[:, [2, 3]]
# Iris-Setosa, Iris-Versicolor, Iris-Virginica
y = iris.target

# Split X and y in into 30 % test data and 70 % into training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0)
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions_(X_combined, y_combined, classifier=tree,
                       test_idx=range(195, 150))
plt.xlabel('petal length (cm)')
plt.ylabel('petal width(cm)')
plt.legend(loc='upper left')
plt.show()

export_graphviz(tree, out_file='tree.dot',
                feature_names=['petal length', 'petal width'])
