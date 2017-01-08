import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10

batch_size = 100

# height x weight
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


def neural_network_model(data):
    hidden_1_layer = {'weights': tf.Variable(
        tf.truncated_normal([784, n_nodes_hl1], stddev=1.0 / float(784))),
        # 'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
        'biases': tf.ones(n_nodes_hl1, dtype='float')}

    hidden_2_layer = {'weights': tf.Variable(
        tf.truncated_normal([n_nodes_hl1, n_nodes_hl2],
                            stddev=1.0 / float(n_nodes_hl1))),
                      # 'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
                      'biases': tf.ones(n_nodes_hl2, dtype='float')}

    hidden_3_layer = {'weights': tf.Variable(
        tf.truncated_normal([n_nodes_hl2, n_nodes_hl3],
                            stddev=1.0 / float(n_nodes_hl2))),
                      # 'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
                      'biases': tf.ones(n_nodes_hl3, dtype='float')}

    output_layer = {'weights': tf.Variable(
        tf.truncated_normal([n_nodes_hl3, n_classes],
                            stddev=1.0 / float(n_nodes_hl3))),
                    # 'biases': tf.Variable(tf.random_normal([n_classes]))}
                    'biases': tf.ones(n_classes, dtype='float')}

    l1 = tf.add(tf.matmul(
        data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(
        l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(
        l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer['weights']),
                    output_layer['biases'])

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # tf.train.write_graph(tf.get_default_graph().as_graph_def(),
        #                      "/tmp", "inference.pbtxt", as_text=True)

        # train the model
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={
                    x: epoch_x, y: epoch_y})
                epoch_loss += c
            print 'Epoch: {}, out of {}. Loss: {:.3f}'.format(
                epoch + 1, hm_epochs, epoch_loss)

            # test the model
            # tf.argmax returns the index of max element
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print 'Accuracy: {}'.format(accuracy.eval({x: mnist.test.images,
                                                       y: mnist.test.labels}))


train_neural_network(x)
