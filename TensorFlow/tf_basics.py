import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg


x = tf.placeholder('float', 3)
y = x * 2

with tf.Session() as session:
    result = session.run(y, feed_dict={x: [1, 2, 3]})
    print(result)

x = tf.placeholder('float', [None, 3])
y = tf.mul(x, 2)

x_data = [[1, 2, 3],
          [4, 5, 6]]

with tf.Session() as session:
    result = session.run(y, feed_dict={x: x_data})
    print(result)


# Load image
filename = "./blue_iris.png"

raw_image_data = mpimg.imread(filename)
print raw_image_data.shape
# Create TensorFlow Variable
# X = tf.Variable(raw_image_data, name='x')
X = tf.constant(raw_image_data)

model = tf.global_variables_initializer()


with tf.Session() as session:
    # Some basic operation
    transpose_op = tf.transpose(X, perm=[1, 0, 2])
    session.run(model)
    result = session.run(transpose_op)
    print result.shape

# plt.imshow(result)
# plt.show()

# set placeholders for training data
x = tf.placeholder('float')
y = tf.placeholder('float')

# Variable w is for storing the values
# it is initialized with starting guesses
w = tf.Variable([1.0, 2.0, 3.0], name='w')
y_model = tf.mul(x, w[0]) + w[1]

# square of differences
error = tf.square(y - y_model)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error)

model = tf.global_variables_initializer()
with tf.Session() as session:
    a = 2
    b = 6
    session.run(model)
    for i in range(1000):
        x_val = np.random.random()
        y_val = x_val * a + b
        session.run(train_op, feed_dict={x: x_val, y: y_val})

    w_val = session.run(w)
    print w_val
    print('Predicted model: {:.3f}x + {:.3f}'.format(w_val[0], w_val[1]))
