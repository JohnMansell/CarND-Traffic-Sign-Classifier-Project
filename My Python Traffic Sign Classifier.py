"""==============================================
        Traffic Sign Classifier
==============================================="""

import pickle
from sklearn.utils import shuffle
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.layers import flatten

'''-----------------------------------------------
        Hyper Parameters
-----------------------------------------------'''

EPOCHS = 50
BATCH_SIZE = 64

mu = 0
sigma = 0.1

rate = 0.001
epsilon = 1.0

dropout = 0.9

'''-----------------------------------------------
        Load Data
-----------------------------------------------'''
# Load Pickled Data
training_file = "train.p"
validation_file = "valid.p"
testing_file = "test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

'''-----------------------------------------------
        Visualize the Data
-----------------------------------------------'''

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(1, 1))
plt.imshow(image)
print(y_train[index])

'''-----------------------------------------------
        Preprocess
-----------------------------------------------'''
# Shuffle
X_train, y_train = shuffle(X_train, y_train)

# Normalize
X_train = (X_train / 255) - 0.5
X_valid = (X_valid / 255) - 0.5
X_test = (X_test / 255) - 0.5

'''-----------------------------------------------
        Implement LeNet
-----------------------------------------------'''


def convolution(x, W, b):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
    conv = tf.nn.bias_add(conv, b)
    conv = tf.nn.elu(conv)
    return conv


def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer

    # Weights
    weights = {
        'w_conv1':  tf.Variable(tf.truncated_normal([5, 5, 3, 6], mean=mu, stddev=sigma)),
        'w_conv2':  tf.Variable(tf.truncated_normal([5, 5, 6, 16], mean=mu, stddev=sigma)),
        'w_fc1':    tf.Variable(tf.truncated_normal([400, 120], mean=mu, stddev=sigma)),
        'w_fc2':    tf.Variable(tf.truncated_normal([120, 84], mean=mu, stddev=sigma)),
        'w_fc3':    tf.Variable(tf.truncated_normal([84, 43], mean=mu, stddev=sigma))
    }

    # Biases
    biases = {
        'b_conv1':  tf.Variable(tf.zeros(6)),
        'b_conv2':  tf.Variable(tf.zeros(16)),
        'b_fc1':    tf.Variable(tf.zeros(120)),
        'b_fc2':    tf.Variable(tf.zeros(84)),
        'b_fc3':    tf.Variable(tf.zeros(43))
    }

    # Conv 1:
    conv1 = convolution(x, weights['w_conv1'], biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    # Conv 2
    conv2 = convolution(conv1, weights['w_conv2'], biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    # Fully Connected 1
    fc0 = flatten(conv2)
    fc1 = tf.matmul(fc0, weights['w_fc1']) + biases['b_fc1']
    fc1 = tf.nn.elu(fc1)

    # Fully Connected 2
    fc2 = tf.matmul(fc1, weights['w_fc2']) + biases['b_fc2']
    fc2 = tf.nn.elu(fc2)

    # Fully Connected 3
    logits = tf.matmul(fc2, weights['w_fc3']) + biases['b_fc3']

    return logits
'''-----------------------------------------------
        Features and Labels
-----------------------------------------------'''
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, None)
one_hot_y = tf.one_hot(y, 43)
'''-----------------------------------------------
        Training Pipeline
-----------------------------------------------'''


logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)
'''-----------------------------------------------
        Model Evaluation
-----------------------------------------------'''
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
'''-----------------------------------------------
        Train the Model
-----------------------------------------------'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        if validation_accuracy >= 0.95:
            break

    saver.save(sess, './lenet')
    print("Model saved")

'''-----------------------------------------------
        Evaluate the Model
-----------------------------------------------'''
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
