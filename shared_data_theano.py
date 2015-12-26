import cPickle
import gzip
import theano
import theano.tensor
import numpy as np


def shared_dataset(data_xy):
    """
    Function that loads a dataset into theano shared variables
    
    Helpful if using GPUs, since the dataset is loaded into the GPU memory as a
    whole. That is because loading data into the GPU memory is very slow,
    copying e.g. a minibatch would slow the performance down.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
    # When storing to GPU it should be floats.
    # But during computation the labes are needed as ints.
    # That's why they are casted.
    return shared_x, theano.tensor.cast(shared_y, 'int32')

f = gzip.open('/Users/alperyeg/Downloads/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
test_set_x, test_set_y = shared_dataset(test_set)
valid_set_x, valid_set_y = shared_dataset(valid_set)
train_set_x, train_set_y = shared_dataset(train_set)

batch_size = 500

data = train_set_x[2 * batch_size:3 * batch_size]
label = train_set_y[2 * batch_size:3 * batch_size]
