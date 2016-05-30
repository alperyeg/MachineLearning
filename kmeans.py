from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Create random dataset
X, y = make_blobs(n_samples=1500, n_features=2, centers=3, cluster_std=0.5,
                  shuffle=True, random_state=0)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.grid()
ax1.set_title('dataset')
ax1.scatter(X[:, 0], X[:, 1], c='white', marker='o', s=50)

# Run kmeans algorithm
# init
km = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-04,
            random_state=0)
# actual clustering
y_km = km.fit_predict(X)

ax2.scatter(X[y_km == 0, 0], X[y_km == 0, 1], s=50, c='lightgreen', marker='o', label='cluster1')
ax2.scatter(X[y_km == 1, 0], X[y_km == 1, 1], s=50, c='orange', marker='o', label='cluster2')
ax2.scatter(X[y_km == 2, 0], X[y_km == 2, 1], s=50, c='lightblue', marker='o', label='cluster3')
ax2.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=100, c='red', label='centroids')
ax2.set_title('kmeans')
ax2.grid()
plt.legend()
plt.tight_layout()
plt.show()

"""
How to choose k?
The intra cluster distances (intertia, or distortion) should be smaller with 
increasing k, since the samples will be closer to their centroids. Using this 
intuition choose the k with the most rapid increase in distortion, which is 
called the elbow method.
"""
fig2 = plt.figure()
distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, n_init=10, max_iter=300, tol=1e-04,
                random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)

plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('clusters')
plt.ylabel('SSE - Distortion')
plt.show()
