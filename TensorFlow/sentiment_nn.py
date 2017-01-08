from sentiment_featureset import create_feature_set_and_labels
import numpy as np
import tensorflow as tf


X_train, y_train, X_test, y_test = create_feature_set_and_labels(
    'pos.txt', 'neg.txt', test_size=0.1)


n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2

batch_size = 100

x = tf.placeholder('float', [None, len(X_train[0])])
y = tf.placeholder('float')


def neural_network_model(data):
    hidden_1_layer = {'weights': tf.Variable(
        tf.truncated_normal([len(X_train[0]), n_nodes_hl1],
                            stddev=1.0)),
                      # 'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
                      'biases': tf.ones(n_nodes_hl1, dtype='float')}

    hidden_2_layer = {'weights': tf.Variable(
        tf.truncated_normal([n_nodes_hl1, n_nodes_hl2],
                            stddev=1.0)),
                      # 'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
                      'biases': tf.ones(n_nodes_hl2, dtype='float')}

    hidden_3_layer = {'weights': tf.Variable(
        tf.truncated_normal([n_nodes_hl2, n_nodes_hl3],
                            stddev=1.0)),
                      # 'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
                      'biases': tf.ones(n_nodes_hl3, dtype='float')}

    output_layer = {'weights': tf.Variable(
        tf.truncated_normal([n_nodes_hl3, n_classes],
                            stddev=1.0)),
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

            i = 0
            while i < len(X_train):
                start = i
                end = i + batch_size
                batch_x = np.array(X_train[start:end])
                batch_y = np.array(y_train[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={
                    x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size
            print 'Epoch: {}, out of {}. Loss: {:.3f}'.format(
                epoch + 1, hm_epochs, epoch_loss)

            # test the model
            # tf.argmax returns the index of max element
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print 'Accuracy: {}'.format(accuracy.eval({x: X_test,
                                                       y: y_test}))


train_neural_network(x)
