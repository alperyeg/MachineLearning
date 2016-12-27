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

plt.plot(range(len(nn.cost_)), nn.cost_)
plt.ylim([0, 2000])
plt.ylabel('cost')
plt.xlabel('Epochs * 50')
plt.tight_layout()
plt.show()
