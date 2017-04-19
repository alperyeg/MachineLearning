"""
Training a generative adversarial network to sample from a 
Gaussian distribution, 1-D normal distribution N(-1, 1)

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Target distribution p_data
mu, sigma = -1, 1
xs = np.linspace(-5, 5, 1000)
# Plot the distribution
# plt.plot(xs, norm.pdf(xs, loc=mu, scale=sigma))
# plt.show()

TRAIN_ITERS=10000
# minibatch size
M = 200


# MLP - for D_pre, D1, D2, G networks
def mlp(inpt, output_dim):
    w1 = tf.get_variable('w0', [inpt.get_shape()[1], 6],
                         initializer=tf.random_normal_initializer())
    b1 = tf.get_variable('b0', [6], initializer=tf.constant_initializer(0.0))
    w2 = tf.get_variable('w1', [6, 5],
                         initializer=tf.random_normal_initializer())
    b2 = tf.get_variable('b1', [5], initializer=tf.constant_initializer(0.0))
    w3 = tf.get_variable('w2', [5, output_dim],
                         initializer=tf.random_normal_initializer())
    b3 = tf.get_variable('b2', [output_dim],
                         initializer=tf.constant_initializer(0.0))
    # network operations
    fc1 = tf.nn.tanh(tf.matmul(inpt, w1) + b1)
    fc2 = tf.nn.tanh(tf.matmul(fc1, w2) + b2)
    fc3 = tf.nn.tanh(tf.matmul(fc2, w3) + b3)
    return fc3, [w1, b1, w2, b2, w3, b3]
