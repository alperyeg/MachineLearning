"""
Training a generative adversarial network to sample from a
Gaussian distribution, 1-D normal distribution
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

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
   Linear transformation

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

    # without the minibatch layer, the discriminator needs an additional layer
    # to have enough capacity to separate the two distributions correctly
    if minibatch_layer:
        h2 = run_minibatch(h1)
    else:
        h2 = tf.tanh(linear(h1, h_dim * 2, scope='d2'))

    h3 = tf.sigmoid(linear(h2, 1, scope='d3'))
    return h3


def run_minibatch(inpt, num_kernels=5, kernel_dim=3):
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
    diffs = tf.expand_dims(activation, axis=3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), axis=0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), axis=2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), axis=2)
    return tf.concat(values=[inpt, minibatch_features], axis=1)


def optimizer(loss, var_list, initial_learning_rate, name='GradientDescent',
              **kwargs):
    """
    
    :param loss:  tensor containing the value to minimize.
    :param var_list:  Optional list or tuple of Variable objects to update 
                      to minimize loss. 
    :param initial_learning_rate: tensor or a floating point value.
    :param name: str 
    :param kwargs: Additional keywords
    :return: An Operation that updates the variables in var_list. 
    """
    if name == 'GradientDescent':
        return _gradient_descent_optimizer(loss, var_list,
                                           initial_learning_rate)
    elif name == 'MomentumOptimizer':
        return _momentum_optimizer(loss, var_list, initial_learning_rate)


def _gradient_descent_optimizer(loss, var_list, initial_learning_rate):
    """
    GradientDescentOptimizer with exponential learning rate decay

    """
    # finding good optimization parameters require some tuning
    decay = 0.95
    num_decay_steps = 150
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase=True
    )
    optimizer_ = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss,
        global_step=batch,
        var_list=var_list
    )
    return optimizer_


def _momentum_optimizer(loss, var_list, initial_learning_rate):
    """
    Momentum Optimizer with exponential learning rate decay
    
    """
    # Apply exponential decay to the learning rate; staircase to use integer
    #  division in a stepwise (=staircase) fashion
    decay = 0.95
    num_decay_steps = 150
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase=True)
    optimizer_ = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                            momentum=0.6).minimize(loss,
                                                                   global_step=batch,
                                                                   var_list=var_list)
    return optimizer_


class GAN(object):
    def __init__(self, data, gen, num_steps, batch_size, minibatch, log_every,
                 hidden_size=4, learning_rate=0.03, anim_path="./"):
        """
        :param data: tensor data
        :param gen: tensor generator net
        :param num_steps: int
        :param batch_size: int
        :param minibatch: bool 
        :param log_every: int
        :param anim_path: string
        """
        self.data = data
        self.gen = gen
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.minibatch = minibatch
        self.log_every = log_every
        self.mlp_hidden_size = hidden_size
        self.anim_path = anim_path
        self.anim_frames = []
        self.learning_rate = learning_rate

        # can use a higher learning rate when not using the minibatch layer
        if self.minibatch:
            self.learning_rate = self.learning_rate / 100.0     # 0.005
            print(
                'minibatch active setting smaller learning rate of: {}'.format(
                    self.learning_rate))

        self._create_model()

    def _create_model(self):
        """
        Creates the model
         
        Does the pre-training and optimization steps, creates also the 
        Generative and Discriminator Network.  
        
        """
        # TODO in optimizing steps try MomentumOptimizer
        with tf.variable_scope('D_pre'):
            self.pre_input = tf.placeholder(tf.float32,
                                            shape=(self.batch_size, 1))
            self.pre_labels = tf.placeholder(tf.float32,
                                             shape=(self.batch_size, 1))
            D_pre = discriminator(self.pre_input, self.mlp_hidden_size,
                                  self.minibatch)
            self.pre_loss = tf.reduce_mean(tf.square(D_pre - self.pre_labels))
            self.pre_opt = optimizer(self.pre_loss, None, self.learning_rate)

        # This defines the generator network - it takes samples from a
        # noise distribution as input, and passes them through an MLP.
        with tf.variable_scope('Gen'):
            self.z = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.G = generator(self.z, self.mlp_hidden_size)

        # The discriminator tries to tell the difference between samples from
        # the true data distribution (self.x) and the generated
        # samples (self.z).
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

        # Define the loss for discriminator and generator networks
        # and create optimizers for both
        self.loss_d = tf.reduce_mean(-tf.log(self.D1) - tf.log(1 - self.D2))
        self.loss_g = tf.reduce_mean(-tf.log(self.D2))

        self.   d_pre_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              scope='D_pre')
        self.d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope='Disc')
        self.g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope='Gen')

        self.opt_d = optimizer(self.loss_d, self.d_params, self.learning_rate)
        self.opt_g = optimizer(self.loss_g, self.g_params, self.learning_rate)

    def train(self):
        """
        Draws samples from the data distribution and the noise distribution,
        and alternates between optimizing the parameters of the discriminator
        and the generator.
        """
        with tf.Session() as session:
            tf.global_variables_initializer().run()

            # pretraining discriminator
            num_pretrain_steps = 1000
            for step in range(num_pretrain_steps):
                d = (np.random.random(self.batch_size) - 0.5) * 10.0
                labels = norm.pdf(d, loc=self.data.mu, scale=self.data.sigma)
                pretrain_loss, _ = session.run([self.pre_loss, self.pre_opt], {
                    self.pre_input: np.reshape(d, (self.batch_size, 1)),
                    self.pre_labels: np.reshape(labels, (self.batch_size, 1))
                })
            self.weightsD = session.run(self.d_pre_params)
            tf.summary.histogram('weightsD', self.weightsD[0])

            # copy weights from pre-training over to new D network
            for i, v in enumerate(self.d_params):
                session.run(tf.assign(v, self.weightsD[i]))
                # session.run(v.assign(self.weightsD[i]))

            for step in range(self.num_steps):
                # update discriminator
                x = self.data.sample(self.batch_size)
                z = self.gen.sample(self.batch_size)
                loss_d, _ = session.run([self.loss_d, self.opt_d], {
                    self.x: np.reshape(x, (self.batch_size, 1)),
                    self.z: np.reshape(z, (self.batch_size, 1))
                })

                # update generator
                z = self.gen.sample(self.batch_size)
                loss_g, _ = session.run([self.loss_g, self.opt_g], {
                    self.z: np.reshape(z, (self.batch_size, 1))
                })

                if step % self.log_every == 0:
                    print('{}: {}\t{}'.format(step, loss_d, loss_g))

                if self.anim_path:
                    self.anim_frames.append(self._samples(session))

            if self.anim_path:
                self._save_animation()
            else:
                self._plot_distributions(session)
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter('logs', session.graph)
            writer.close()

    def _samples(self, session, num_points=10000, num_bins=100):
        """
        Return a tuple (db, pd, pg), where db is the current decision
        boundary, pd is a histogram of samples from the data distribution,
        and pg is a histogram of generated samples.
        """
        xs = np.linspace(-self.gen.range, self.gen.range, num_points)
        bins = np.linspace(-self.gen.range, self.gen.range, num_bins)

        # decision boundary
        db = np.zeros((num_points, 1))
        for i in range(num_points // self.batch_size):
            db[self.batch_size * i:self.batch_size * (i + 1)] = session.run(self.D1, {
                self.x: np.reshape(
                    xs[self.batch_size * i:self.batch_size * (i + 1)],
                    (self.batch_size, 1)
                )
            })

        # data distribution
        d = self.data.sample(num_points)
        pd, _ = np.histogram(d, bins=bins, density=True)

        # generated samples
        zs = np.linspace(-self.gen.range, self.gen.range, num_points)
        g = np.zeros((num_points, 1))
        for i in range(num_points // self.batch_size):
            g[self.batch_size * i:self.batch_size * (i + 1)] = session.run(self.G, {
                self.z: np.reshape(
                    zs[self.batch_size * i:self.batch_size * (i + 1)],
                    (self.batch_size, 1)
                )
            })
        pg, _ = np.histogram(g, bins=bins, density=True)

        return db, pd, pg

    def _plot_distributions(self, session):
        db, pd, pg = self._samples(session)
        db_x = np.linspace(-self.gen.range, self.gen.range, len(db))
        p_x = np.linspace(-self.gen.range, self.gen.range, len(pd))
        f, ax = plt.subplots(1)
        ax.plot(db_x, db, label='decision boundary')
        ax.set_ylim(0, 1)
        plt.plot(p_x, pd, label='real data')
        plt.plot(p_x, pg, label='generated data')
        plt.title('1D Generative Adversarial Network')
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        plt.legend()
        plt.savefig('fig1.png', format='png')
        plt.show()

    def _save_animation(self):
        f, ax = plt.subplots(figsize=(6, 4))
        f.suptitle('1D Generative Adversarial Network', fontsize=15)
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        ax.set_xlim(-6, 6)
        ax.set_ylim(0, 1.4)
        line_db, = ax.plot([], [], label='decision boundary')
        line_pd, = ax.plot([], [], label='real data')
        line_pg, = ax.plot([], [], label='generated data')
        frame_number = ax.text(
            0.02,
            0.95,
            '',
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes
        )
        ax.legend()

        db, pd, _ = self.anim_frames[0]
        db_x = np.linspace(-self.gen.range, self.gen.range, len(db))
        p_x = np.linspace(-self.gen.range, self.gen.range, len(pd))

        def init():
            line_db.set_data([], [])
            line_pd.set_data([], [])
            line_pg.set_data([], [])
            frame_number.set_text('')
            return line_db, line_pd, line_pg, frame_number

        def animate(i):
            frame_number.set_text(
                'Frame: {}/{}'.format(i, len(self.anim_frames))
            )
            db, pd, pg = self.anim_frames[i]
            line_db.set_data(db_x, db)
            line_pd.set_data(p_x, pd)
            line_pg.set_data(p_x, pg)
            return (line_db, line_pd, line_pg, frame_number)

        anim = animation.FuncAnimation(
            f,
            animate,
            init_func=init,
            frames=len(self.anim_frames),
            blit=True
        )
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, bitrate=1800)
        # anim.save(self.anim_path, fps=30, extra_args=['-vcodec', 'libx264'])
        anim.save(self.anim_path, writer=writer)


def main(**kwargs):
    model = GAN(DataDistribution(),
                GeneratorDistribution(range_=8),
                num_steps=kwargs['num_steps'],
                batch_size=kwargs['batch_size'],
                minibatch=kwargs['minibatch'],
                log_every=kwargs['log_every'],
                anim_path=kwargs['anim_path']
                )
    model.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=1200,
                        help='the number of training steps to take')
    parser.add_argument('--batch-size', type=int, default=12,
                        help='the batch size')
    parser.add_argument('--minibatch', type=bool, default=False,
                        help='use minibatch discrimination')
    parser.add_argument('--log-every', type=int, default=10,
                        help='print loss after this many steps')
    parser.add_argument('--anim', type=str, default=None,
                        help='name of the output animation file (default: none)')
    return parser.parse_args()

if __name__ == '__main__':
    num_steps = 1200
    batch_size = 12
    minibatch = False
    log_every = 10
    anim_path = ""
    main(num_steps=num_steps, batch_size=batch_size, minibatch=minibatch,
         log_every=log_every, anim_path=anim_path)