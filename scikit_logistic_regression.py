from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from decision_regions import plot_decision_regions_
import numpy as np
import matplotlib.pyplot as plt


# load iris dataset
iris = datasets.load_iris()
# last two features of iris dataset
X = iris.data[:, [2, 3]]
# Iris-Setosa, Iris-Versicolor, Iris-Virginica
y = iris.target

# Split X and y in into 30 % test data and 70 % into trainining data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0)
sc = StandardScaler()
# Estimate mean and std using fit method
sc.fit(X_train)
# Standarize training and test data using mean and std via transform method
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions_(X=X_combined_std, y=y_combined,
                       classifier=lr, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
