#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 09:11:07 2019

@author: jas10022
"""

import numpy as np
import pandas as pd

data = np.load("mnist.npz")
X = data["X"]
labels = data["y"]

# Import the `transform` module from `skimage`
from skimage import transform 

# Rescale the images in the `images` array
images28 = [transform.resize(X, (28, 28)) for image in X]

#split the data into test and train data


#setup the neural network
import tensorflow as tf 

x = tf.placeholder(dtype = tf.float32, shape = [none, 28, 28])
y = tf.placeholder(dtype = tf.int8, shape = [none])

images_flat = tf.contrib.layers.flatten(x)

logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))

train_op = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)

correct_pred = tf.argmax(logits, 1)

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# run the neural network
tf.set_random_seed(28394)

session = tf.Session()

session.run(tf.global_variables_initializer())

for i in range(70000):
        print('EPOCH', i)
        _, accuracy_val = session.run([train_op, accuracy], feed_dict={x: images28, y: labels})
        if i % 10 == 0:
            print("Loss: ", loss)
        print('DONE WITH EPOCH')

# run test data

prdicted = session.run([correct_pred],feed_dict = {x:test_images28})[0]

match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])

accuracy = match_count / len(test_labels)

# Print the accuracy
print("Accuracy: {:.3f}".format(accuracy))

session.close()
























