import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from matplotlib import animation
from scipy.stats import norm
from six.moves import range

sns.set(color_codes=True)

seed = 1234
np.random.seed(seed)
tf.set_random_seed(seed)


class DataDistribution(object):
    def __init__(self):
        """
       Real data distribution, a simple Gaussian with mean 4 and standard
       deviation of 0.5. It has a sample function that returns a given
       number of samples (sorted by value) from the distribution.
        """
        self.mu = 4
        self.sigma = 0.5

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples


class GeneratorDistribution(object):
    def __init__(self, range_):
        """
         Generator input noise distribution (with a similar sample function).
         A stratified sampling approach for the generator input noise - the
         samples are first generated uniformly over a specified range,
         and then randomly perturbed.
        """
        self.range = range_

    def sample(self, N):
        return np.linspace(-self.range, self.range, N) + np.random.random(N)\
            * 0.01


def linear(inpt, output_dim, scope=None, stddev=1.0):
    """
    Creates a multilayer perceptron

    :param inpt: data
    :param output_dim: hidden layers
    :param scope: name
    :param stddev: standard deviation
    :return: tensor of type input
    """
    normal = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable('w', [inpt.get_shape()[1], output_dim],
                            initializer=normal)
        b = tf.get_variable('b', [output_dim], initializer=const)
        return tf.matmul(inpt, w) + b


# The generator and discriminator networks are quite simple.
# The generator is a linear transformation passed through a non-linearity
# (a softplus function), followed by another linear transformation.
def generator(inpt, h_dim):
    h0 = tf.nn.softplus(linear(inpt, h_dim, 'g0'))
    h1 = linear(h0, 1, 'g1')
    return h1


# Make sure that the discriminator is more powerful than the generator,
# as otherwise it did not have sufficient capacity to learn to be able to
# distinguish accurately between generated and real samples.
# So make it a deeper neural network, with a larger number of dimensions.
# It uses tanh nonlinearities in all layers except the final one, which is
# a sigmoid (the output of which is interpreted as a probability).
def discriminator(inpt, h_dim, minibatch_layer=True):
    h0 = tf.tanh(linear(inpt, h_dim * 2, scope='d0'))
    h1 = tf.tanh(linear(h0, h_dim * 2, scope='d1'))
    if minibatch_layer:
        h2 = minibatch(h1)
    else:
        h2 = tf.tanh(linear(h1, h_dim * 2, scope='d2'))
    h3 = tf.sigmoid(linear(h2, 1, scope='h3'))
    return h3


def minibatch(inpt, num_kernels=5, kernel_dim=3):
    """
    * Take the output of some intermediate layer of the discriminator.
    * Multiply it by a 3D tensor to produce a matrix (of size num_kernels x 
    kernel_dim in the code below).
    * Compute the L1-distance between rows in this matrix across all samples 
    in a batch, and then apply a negative exponential.
    * The minibatch features for a sample are then the sum of these 
    exponentiated distances.
    * Concatenate the original input to the minibatch layer (the output of 
    the previous discriminator layer) with the newly created minibatch 
    features, and pass this as input to the next layer of the discriminator.
    
    :param inpt: 
    :param num_kernels: 
    :param kernel_dim: 
    :return: 
    """
    x = linear(inpt, num_kernels * kernel_dim, scope='minibatch', stddev=0.02)
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - tf.expand_dims(
        tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat(1, [inpt, minibatch_features])


def optimizer(loss, var_list, initial_learning_rate):
    """
    GradientDescentOptimizer with exponential learning rate decay

    """
    # finding good optimization parameters require some tuning
    decay = 0.05
    num_decay_steps = 150
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(initial_learning_rate, batch,
                                               num_decay_steps, decay,
                                               staircase=True)
    # TODO try the MomentumOptimizer
    optimizer_ = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss,
        global_step=batch,
        var_list=var_list)
    return optimizer_


class GAN(object):
    def __init__(self, data, gen, num_steps, batch_size, minibatch,
                 log_every, anim_path):
        """
        
        :param data: tensor data
        :param gen: tensor generator net
        :param num_steps: int
        :param batch_size: int
        :param minibatch: bool 
        :param log_every: bool
        :param anim_path: string
        """
        self.data = data
        self.gen = gen
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.minibatch = minibatch
        self.log_every = log_every
        self.mlp_hidden_size = 4
        self.anim_path = anim_path
        self.anim_frames = []
        self.learning_rate = 0.03

        # can use a higher learning rate when not using the minibatch layer
        if self.minibatch:
            self.learning_rate = 0.005

        self._create_model()

    def _create_model(self):
        # In order to make sure that the discriminator is providing useful
        # gradient information to the generator from the start,
        # pretrain the discriminator using a maximum likelihood objective.
        # Define the network for this pretraining step scoped as D_pre.
        with tf.variable_scope('D_pre'):
            self.pre_input = tf.placeholder(
                tf.float32, shape=(self.batch_size, 1))
            self.pre_labels = tf.placeholder(
                tf.float32, shape=(self.batch_size, 1))
            D_pre = discriminator(
                self.pre_input, self.mlp_hidden_size, self.minibatch)
            self.pre_loss = tf.reduce_mean(tf.square(D_pre - self.pre_labels))
            self.pre_opt = optimizer(self.pre_loss, None, self.learning_rate)

        # This defines the generator network - it takes samples from a noise
        # distribution as input, and passes them through an MLP.
        with tf.variable_scope('Gen'):
            self.z = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.G = generator(self.z, self.mlp_hidden_size)

        # The discriminator tries to tell the difference between samples from
        # the true data distribution (self.x) and the generated samples
        # (self.z).
        #
        # Here we create two copies of the discriminator network
        # (that share parameters), as you cannot use the same network with
        # different inputs in TensorFlow.
        with tf.variable_scope('Disc') as scope:
            self.x = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.D1 = discriminator(self.x, self.mlp_hidden_size,
                                    self.minibatch)
            scope.reuse_variables()
            self.D2 = discriminator(self.G, self.mlp_hidden_size,
                                    self.minibatch)

        # Define the loss for discriminator and generator networks and create
        # optimizers for both
        self.loss_d = tf.reduce_mean(-tf.log(self.D1) - tf.log(1 - self.D2))
        self.loss_g = tf.reduce_mean(-tf.log(self.D2))

        self.d_pre_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_pre')
        self.d_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='Disc')
        self.g_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='Gen')

        self.opt_d = optimizer(self.loss_d, self.d_params, self.learning_rate)
        self.opt_g = optimizer(self.loss_g, self.g_params, self.learning_rate)

    def train(self):
        """
        Draws samples from the data distribution and the noise distribution,
        and alternates between optimizing the parameters of the discriminator
        and the generator.
        """
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
        # pretraining step
        num_pretrain_steps = 1000
        for step in range(num_pretrain_steps):
            d = (np.random.random(self.batch_size) - 0.5) * 10.0
            labels = norm.pdf(d, loc=self.data.mu, scale=self.data.sigma)
            pretrain_loss, _ = sess.run([self.pre_loss, self.pre_opt], {
                self.pre_input: np.reshape(d, (self.batch_size, 1)),
                self.pre_labels: np.reshape(labels, (self.batch_size, 1))})
        self.weightsD = sess.run(self.d_pre_params)

        # copy weights from pre-training over to new D network
        for i, v in enumerate(self.d_params):
            sess.run(v.assign(self.weightsD[i]))

        for step in range(self.num_steps):
            # update discriminator
            x = self.data.sample(self.batch_size)
            z = self.gen.sample(self.batch_size)
            loss_d, _ = sess.run([self.loss_d, self.opt_d], {
                self.x: np.reshape(x, (self.batch_size, 1)),
                self.z: np.reshape(z, (self.batch_size, 1))
            })
            # update generator
            z = self.gen.sample(self.batch_size)
            loss_g, _ = sess.run([self.loss_g, self.opt_g], {
                self.z: np.reshape(z, (self.batch_size, 1))
            })

            if step % self.log_every == 0:
                print('{}: {}\t{}'.format(step, loss_d, loss_g))

            if self.anim_path:
                self.anim_frames.append(self._samples(sess))
        if self.anim_path:
            self._save_animation()
        else:
            self._plot_distributions(sess)

    def _samples(self, session, num_points=1000, num_bins=100):
        pass

    def _plot_distributions(self, session):
        pass

    def _save_animation(self):
        pass