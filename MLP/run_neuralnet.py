from neuralnet import NeuralNetMLP
import os
import struct
import numpy as np
import matplotlib.pyplot as plt


def load_mnist(path, kind='train'):
    """
    Load mnist data from `path`
    """
    labels_path = os.path.join(path, '{}-labels-idx1-ubyte'.format(kind))
    images_path = os.path.join(path, '{}-images-idx3-ubyte'.format(kind))

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels),
                                                              cols * rows)

    return images, labels


X_train, y_train = load_mnist('./', kind='train')
print('Rows: {}, Columns: {}'.format(X_train.shape[0], X_train.shape[1]))
X_test, y_test = load_mnist('./', kind='t10k')
# print('Rows: {}, Columns: {}'.format(X_test.shape[0], X_test.shape[1]))

nn = NeuralNetMLP(n_output=10, n_features=X_train.shape[1], n_hidden=50,
                  l2=0.1, l1=0.0, epochs=1000, eta=0.001, alpha=0.001,
                  decrease_const=0.00001, shuffle=True, minibatches=50,
                  random_state=1)
nn.fit(X_train, y_train, print_progress=True)


def plot_cost(net):
    """
    Plot costs pro epoch.
    Plot every 50th step to account for 50 mini-batches (50 x 1000 epochs).
    """
    plt.plot(range(len(net.cost_)), net.cost_)
    plt.ylim([0, 2000])
    plt.ylabel('cost')
    plt.xlabel('Epochs * 50')
    plt.tight_layout()
    plt.show()


def plot_cost_batches(net):
    """
    Plots a smooth version of the cost function by averaging over the
    mini-batch interval.
    """
    batches = np.array_split(range(len(net.cost_)), 1000)
    cost_ary = np.array(net.cost_)
    cost_avgs = [np.mean(cost_ary[i]) for i in batches]
    plt.plot(range(len(cost_avgs)), cost_avgs, color='red')
    plt.ylim([0, 2000])
    plt.ylabel('cost')
    plt.xlabel('Epochs * 50')
    plt.tight_layout()
    plt.show()


# Plot costs
plot_cost(nn)
plot_cost_batches(nn)

# evaluate the performance
y_train_pred = nn.predict(X_train)
acc = np.sum(y_train == y_train_pred, axis=0) / float(X_train.shape[0])
print('Training accuracy: {:.2%}'.format(acc * 100))

# evaluate the performance on test set
y_test_pred = nn.predict(X_test)
acc = np.sum(y_test == y_test_pred, axis=0) / float(X_test.shape[0])
print('Training accuracy: {:.2%}'.format(acc * 100))

miscl_img = X_test[y_test != y_test_pred][:25]
correct_lab = y_test[y_test != y_test_pred][:25]
miscl_lab = y_test_pred[y_test != y_test_pred][:25]

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title('{0}) t: {1} p: {2}'.format(i+1, correct_lab[i], miscl_lab[
        i]))
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
