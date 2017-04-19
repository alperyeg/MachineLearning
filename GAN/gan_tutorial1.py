"""
Training a generative adversarial network to sample from a 
Gaussian distribution, 1-D normal distribution N(-1, 1)

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Set seed, for reproducibility
tf.set_random_seed(1234)

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
    # tf.get_variable if you want to share variables, otherwise tf.Variable
    # is also fine
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


# optimize networks
def momentum_optimizer(loss, var_list):
    batch = tf.Variable(0)
    # Apply exponential decay to the learning rate, staircase to use integer
    #  division in a stepwise (=staircase) fashion
    learning_rate = tf.train.exponential_decay(0.001, batch, TRAIN_ITERS //
                                               4, 0.95, staircase=True)
    # optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(
    #     loss, global_step=batch,var_list=var_list)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                           momentum=0.6).minimize(loss,
                                                                  global_step=batch,
                                                                  var_list=var_list)
    return optimizer

# pre train decision surface
with tf.variable_scope("D_pre"):
    input_node = tf.placeholder(tf.float32, shape=(M, 1))
    train_labels = tf.placeholder(tf.float32, shape=(M, 1))
    D, theta = mlp(input_node, 1)
    loss = tf.reduce_mean(tf.square(D - train_labels))
    optimizer = momentum_optimizer(loss, None)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()


# Plot the decision surface
def plot_decision_surface(D, inpt_node):
    f, ax = plt.subplots(1)
    # p_data
    xs = np.linspace(-5, 5, 1000)
    ax.plot(xs, norm.pdf(xs, loc=mu, scale=sigma), label='p_data')
    # decision boundary
    r = 1000    # resolution = #points
    xs = np.linspace(-5,5, r)
    # decision surface
    ds = np.zeros((r, 1))
    for i in range(r/M):
        x = np.reshape(xs[M*i:M*(i+1)],(M,1))
        ds[M * i:M * (i + 1)] = sess.run(D, {inpt_node: x})

    ax.plot(xs, ds, label='decision boundary')
    ax.set_ylim(0, 1.1)
    plt.legend()
    plt.title('Initial Decision Boundary')
    plt.show()


plot_decision_surface(D, input_node)

lh = np.zeros(1000)
for i in range(1000):
    # d=np.random.normal(mu,sigma,M)
    d = (np.random.random(M) - 0.5) * 10.0
    labels = norm.pdf(d, loc=mu, scale=sigma)
    lh[i], _ = sess.run([loss, optimizer], {input_node: np.reshape(d, (M, 1)),
                                            train_labels: np.reshape(labels,
                                                                     (M, 1))})
# training loss
plt.plot(lh)
plt.title('Training Loss')
plot_decision_surface(D,input_node)
# copy the learned weights over into a tmp array
weightsD = sess.run(theta)
# close the pre-training session
sess.close()
plt.show()

# TODO: Building the generative net