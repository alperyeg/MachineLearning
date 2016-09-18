from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

feature_dict = {i: label for i, label in zip(range(4),
                ('sepal length in cm',
                 'sepal width in cm',
                 'petal length in cm',
                 'petal width in cm', ))}
df = pd.io.parsers.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None, sep=',')
df.columns = [l for i, l in sorted(feature_dict.items())] + ['class label']
df.dropna(how='all', inplace=True)
df.tail()

X = df[[0, 1, 2, 3]].values
y = df['class label'].values

enc = LabelEncoder()
label_encoder = enc.fit(y)
y = label_encoder.transform(y) + 1
label_dict = {1: 'Setosa', 2: 'Versicolor', 3: 'Virginica'}


def plot_hist(X, y, label_dict):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))

    for ax, cnt in zip(axes.ravel(), range(4)):
        # set bin sizes    
        min_b = math.floor(np.min(X[:, cnt]))
        max_b = math.floor(np.max(X[:, cnt]))
        bins = np.linspace(min_b, max_b, 25)

        for lab, col in zip(range(1, 4), ('blue', 'red', 'green')):
            ax.hist(X[y == lab, cnt], color=col,
                    label='class {}'.format(label_dict[lab]),
                    bins=bins, alpha=0.5)
            ylims = ax.get_ylim()

            # plot annotation
            leg = ax.legend(loc='upper right', fancybox=True, fontsize=8)
            leg.get_frame().set_alpha(0.5)
            ax.set_ylim([0, max(ylims)+2])
            ax.set_xlabel(feature_dict[cnt])
            ax.set_title('Iris histogram #{}'.format(str(cnt + 1)))
            
            # hide axis ticks
            ax.tick_params(axis="both", which="both", bottom="off", top="off",
                           labelbottom="on", left="off", right="off",
                           labelleft="on")

            # remove axis spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    axes[0][0].set_ylabel('count')
    axes[1][0].set_ylabel('count')

    fig.tight_layout()
    plt.show()

# plot_hist(X, y, label_dict)

np.set_printoptions(precision=4)

mean_vectors = []
for cl in range(1, 4):
    mean_vectors.append(np.mean(X[y == cl], axis=0))
    print ('Mean vector of class {}: {}'.format(cl, mean_vectors[cl - 1]))

# Calculte the within-class scatter matrix Sw
S_w = np.zeros((4, 4))
for cl, mv in zip(range(1, 4), mean_vectors):
    # scatter matrix for every class
    class_sc_mat = np.zeros((4, 4))
    for row in X[y == cl]:
        # make column vectors
        row, mv = row.reshape(4, 1), mv.reshape(4, 1)
        class_sc_mat += (row - mv).dot((row - mv).T)
    S_w += class_sc_mat
print('within-class Scatter Matrix: \n {}'.format(S_w))

overall_mean = np.mean(X, axis=0)

S_b = np.zeros((4, 4))
for i, mean_vec in enumerate((mean_vectors)):
    n = X[y == i + 1, :].shape[0]
    # make column vector
    mean_vec = mean_vec.reshape(4, 1)
    overall_mean = overall_mean.reshape(4, 1)
    S_b += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
print('between-class Scatter Matrix: \n {}'.format(S_b))

eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_w).dot(S_b))

for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:, i].reshape(4, 1)
    print('\nEigenvector {}: \n{}'.format(i+1, eigvec_sc.real))
    print('Eigenvalue {:}: {:.2e}'.format(i+1, eig_vals[i].real))

# Make a list of (eigenvalues, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i
             in range(len(eig_vals))]
# Sort them from high to low
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
print('Eigenvalues in decreasing order:\n')
for i in eig_pairs:
    print(i[0])
print('Variance explained \n')
eigv_sum = np.sum(eig_vals)
for i, j in enumerate(eig_pairs):
    print('eigenvalue {0:}: {1:.2%}'.format(i + 1, (j[0] / eigv_sum).real))

W = np.hstack((eig_pairs[0][1].reshape(4, 1), eig_pairs[1][1].reshape(4, 1)))
print('Matrix W:\n {}'.format(W.real))

X_lda = X.dot(W)

def plot_step_lda():
    ax = plt.subplot(111)
    for label, marker, color in zip(
            range(1, 4), ('^', 's', 'o'), ('blue', 'red', 'green')):

        plt.scatter(x=X_lda[:, 0].real[y == label],
                    y=X_lda[:, 1].real[y == label],
                    marker=marker,
                    color=color,
                    alpha=0.5,
                    label=label_dict[label])

    plt.xlabel('LD1')
    plt.ylabel('LD2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('LDA: Iris projection onto the first 2 linear discriminants')

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.grid()
    plt.tight_layout
    plt.show()

plot_step_lda()
