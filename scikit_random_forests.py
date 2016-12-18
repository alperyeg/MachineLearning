from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
forest = RandomForestClassifier(criterion='entropy', n_estimators=10,
                                random_state=1, n_jobs=4)
forest.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions_(X_combined, y_combined, classifier=forest,
                       test_idx=range(105, 150))
plt.xlabel('petal length (cm)')
plt.ylabel('petal width(cm)')
plt.legend(loc='upper left')
plt.show()
