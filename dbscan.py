from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
plt.scatter(X[:, 0], X[:, 1])
plt.show()

# Run kmeans
km = KMeans(n_clusters=2, init='random', n_init=10, max_iter=300, tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

# Run dbscan
db = DBSCAN(eps=0.2, min_samples=5, algorithm='auto', metric='euclidean')
y_db = db.fit_predict(X)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.scatter(X[y_km == 0, 0], X[y_km == 0, 1], s=40, c='lightgreen', marker='o', label='cluster1')
ax1.scatter(X[y_km == 1, 0], X[y_km == 1, 1], s=40, c='lightblue', marker='s', label='cluster2')
ax1.set_title('k-means')

ax2.scatter(X[y_db == 0, 0], X[y_db == 0, 1], s=40, c='lightgreen', marker='o', label='cluster1')
ax2.scatter(X[y_db == 1, 0], X[y_db == 1, 1], s=40, c='lightblue', marker='s', label='cluster2')
ax2.set_title('dbscan')
plt.legend()
plt.show()
