#!/usr/bin/env python
# -*- coding: utf-8 -*-

# modified from https://github.com/wolfib/image-classification-CIFAR10-tf/blob/master/softmax.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
import data_helpers

beginTime = time.time()

# Parameter definitions
num_features = data_helpers.SIZE[0] * data_helpers.SIZE[1] * 3
batch_size = 100
learning_rate = 0.005
max_steps = 1000

# Uncommenting this line removes randomness
# np.random.seed(1)

# Prepare data
X_train, Y_train, X_test, Y_test = data_helpers.load_data()

# -----------------------------------------------------------------------------
# Prepare the TensorFlow graph
# (We're only defining the graph here, no actual calculations taking place)
# -----------------------------------------------------------------------------

# Define input placeholders
images_placeholder = tf.placeholder(tf.float32, shape=[None, num_features])
labels_placeholder = tf.placeholder(tf.int64, shape=[None])

# Define variables (these are the values we want to optimize)
weights = tf.Variable(tf.zeros([num_features, 2]))
biases = tf.Variable(tf.zeros([2]))

# Define the classifier's result
logits = tf.matmul(images_placeholder, weights) + biases

# Define the loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
  labels=labels_placeholder))

# Define the training operation
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Operation comparing prediction with true label
correct_prediction = tf.equal(tf.argmax(logits, 1), labels_placeholder)

# Operation calculating the accuracy of our predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# -----------------------------------------------------------------------------
# Run the TensorFlow graph
# -----------------------------------------------------------------------------

with tf.Session() as sess:
  # Initialize variables
  sess.run(tf.global_variables_initializer())

  # Repeat max_steps times
  for i in range(max_steps):

    # Generate input data batch
    indices = np.random.choice(X_train.shape[0], batch_size)
    images_batch = X_train[indices]
    labels_batch = Y_train[indices]

    # Periodically print out the model's current accuracy
    if i % 100 == 0:
      train_accuracy = sess.run(accuracy, feed_dict={
        images_placeholder: images_batch, labels_placeholder: labels_batch})
      print('Step {:5d}: training accuracy {:g}'.format(i, train_accuracy))

    # Perform a single training step
    sess.run(train_step, feed_dict={images_placeholder: images_batch,
      labels_placeholder: labels_batch})

  # After finishing the training, evaluate on the test set
  test_accuracy = sess.run(accuracy, feed_dict={
    images_placeholder: X_test,
    labels_placeholder: Y_test})
  print('Test accuracy {:g}'.format(test_accuracy))

endTime = time.time()
print('Total time: {:5.2f}s'.format(endTime - beginTime))
