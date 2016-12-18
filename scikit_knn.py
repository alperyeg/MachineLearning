from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
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

sc = StandardScaler()
# Estimate mean and std using fit method
sc.fit(X_train)
# Standardize training and test data using mean and std via transform method
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)

plot_decision_regions_(X_combined_std, y_combined, classifier=knn,
                       test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.show()
