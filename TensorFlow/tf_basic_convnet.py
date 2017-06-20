# https://www.tensorflow.org/tutorials/layers

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib


tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    """

    :param features:
    :param labels:
    :param mode:
    :return:
    """
    # Input layer
    '''
    * shape of [batch_size, image_width, image_height, channels]
    * batch_size: Size of the subset of examples to use when performing,
      -1 specifies that this dimension should be dynamically computed based on 
      the number of input values in features, holding the size of all other 
      dimensions constant
    * gradient descent during training.
    * image_width: Width of the example images
    * image_height: Height of the example images
    * channels: Number of color channels in the example images. 
      For color images, the number of channels is 3 (red, green, blue). 
      For monochrome images, there is just 1 channel (black)
    '''
    input_layer = tf.reshape(features, [-1, 28, 28, 1])

    # Convolutional Layer #1
    # apply 32 5x5 filters to the input layer, with a ReLU activation function
    # output is a tensor of size [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5],
                             padding='same', activation=tf.nn.relu)

    # Pooling Layer 1
    # [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    # [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5],
                             padding='same', activation=tf.nn.relu)
    # [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense layer
    # flatten the feature map (pool2) to shape [batch_size, features]
    '''
    Each example has 7 (pool2 width) * 7 (pool2 height) * 64 (pool2 channels) 
    features, so we want the features dimension to have a value of 7 * 7 * 64 
    (3136 in total)
    '''
    # [batch_size, 3136]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024,
                            activation=tf.nn.relu)
    #  rate: 40% of the elements will be randomly dropped out during training
    # [batch_size, 1024]
    dropout = tf.layers.dropout(inputs=dense, rate=0.4,
                                training=mode == learn.ModeKeys.TRAIN)

    # Logits layer
    # 10 neurons for each number, will have probabilities
    # [batch, 10]
    logits = tf.layers.dense(inputs=dropout, units=10)

    loss = None
    train_op = None

    # Calculate loss (for Train and Eval)
    if mode != learn.ModeKeys.INFER:
        # First convert to binary representation
        onehot_labes = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labes,
                                               logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(loss=loss,
                                                   global_step=tf.contrib.framework.get_global_step(),
                                                   learning_rate=0.001,
                                                   optimizer="SGD")

    # Generate Predictions
    '''
    * classes: maximum of logits
    * probabilities: probabilities from logits layer by applying softmax activation
    '''
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def main(unused_argv):
    # Load training and eval data
    mnist = learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images   # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    # Create the Estimator
    mnist_classifier = learn.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
    # Set up logging for predictions
    # dict of the tensors log in tensors_to_log
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                              every_n_iter=50)
    # Train the model
    mnist_classifier.fit(
        x=train_data,
        y=train_labels,
        batch_size=100,
        steps=20000,
        monitors=[logging_hook])

    # Configure the accuracy metric for evaluation
    '''
    * metric_fn: The function that calculates and returns the value of the metric.
      Here, we can use the predefined accuracy function in the tf.metrics module.
    * prediction_key: The key of the tensor that contains the predictions 
      returned by the model function, prediction key is "classes", 
      see predictions dictionary 
    '''
    metrics = {
        "accuracy":
            learn.MetricSpec(
                metric_fn=tf.metrics.accuracy, prediction_key="classes"),
    }
    # Evaluate the model and print results
    eval_results = mnist_classifier.evaluate(
        x=eval_data, y=eval_labels, metrics=metrics)
    print(eval_results)


if __name__ == '__main__':
    tf.app.run()
