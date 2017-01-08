import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')

np.random.seed(1)
tf.set_random_seed(1)

# create dataset, something like y = x * 0.1 + 0.3 + noise
x_data = np.random.rand(100).astype(np.float32)
noise = np.random.normal(scale=0.01, size=len(x_data))
y_data = x_data * 0.1 + 0.3 + noise

# plt.plot(x_data, y_data, '.')
# plt.show()

# Build inference graph
# y_data = W * x_data + b
W = tf.Variable(tf.random_uniform(shape=[1], minval=0.0, maxval=1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# Build training graph
loss = tf.reduce_mean(tf.square(y - y_data))
train = tf.train.GradientDescentOptimizer(0.3).minimize(loss)
init = tf.global_variables_initializer()

# Uncomment to see how the graph looks like
# print(tf.get_default_graph().as_graph_def())

# create session and launch the graph
with tf.Session() as session:
    session.run(init)
    y_initial_values = session.run(y)
    # print(session.run([W, b])) # initial values
    for step in range(1000):
        session.run(train)
        # Uncomment the following two lines to watch training
        # happen real time.
        # if step % 20 == 0:
            # print(step, session.run([W, b]))
        # print(session.run([W, b]))
    plt.plot(x_data, y_data, '.', label="target_values")
    plt.plot(x_data, y_initial_values, ".", label="initial_values")
    plt.plot(x_data, session.run(y), ".", label="trained_values")
    plt.legend()
    plt.ylim(0, 1.0)
    plt.show()