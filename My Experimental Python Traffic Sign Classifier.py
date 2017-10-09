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
from skimage.transform import rotate

'''-----------------------------------------------
        Hyper Parameters
-----------------------------------------------'''

EPOCHS = 50
BATCH_SIZE = 64

mu = 0
sigma = 0.1

rate = 0.001

keep_prob = 0.9

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

print(y_train[index])

'''-----------------------------------------------
        Preprocess
-----------------------------------------------'''

num_examples = len(X_train)
X_train, y_train = shuffle(X_train, y_train)

print('done')

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
        'w_conv1':  tf.Variable(tf.truncated_normal([1, 1, 3, 1], mean=mu, stddev=sigma)),
        'w_conv2':  tf.Variable(tf.truncated_normal([5, 5, 1, 1], mean=mu, stddev=sigma)),
        'w_conv3':  tf.Variable(tf.truncated_normal([3, 3, 1, 1], mean=mu, stddev=sigma)),
        'w_conv4':  tf.Variable(tf.truncated_normal([1, 1, 3, 1], mean=mu, stddev=sigma)),

        'w_ins1':   tf.Variable(tf.truncated_normal([1024, 800], mean=mu, stddev=sigma)),
        'w_ins2':   tf.Variable(tf.truncated_normal([784, 800], mean=mu, stddev=sigma)),
        'w_ins3':   tf.Variable(tf.truncated_normal([900, 800], mean=mu, stddev=sigma)),
        'w_ins4':   tf.Variable(tf.truncated_normal([256, 800], mean=mu, stddev=sigma)),

        'full_1': tf.Variable(tf.truncated_normal([3200, 1600], mean=mu, stddev=sigma)),
        'full_2': tf.Variable(tf.truncated_normal([1600, 800], mean=mu, stddev=sigma)),
        'full_3': tf.Variable(tf.truncated_normal([800, 200], mean=mu, stddev=sigma)),
        'full_4': tf.Variable(tf.truncated_normal([200, 43], mean=mu, stddev=sigma))
    }

    # Biases
    biases = {
        'b_conv1': tf.Variable(tf.zeros(1)),
        'b_conv2': tf.Variable(tf.zeros(1)),
        'b_conv3': tf.Variable(tf.zeros(1)),

        'b_max_p': tf.Variable(tf.zeros(1)),
        'b_conv4': tf.Variable(tf.zeros(1)),

        'b_ins1': tf.Variable(tf.zeros(1)),
        'b_ins2': tf.Variable(tf.zeros(1)),
        'b_ins3': tf.Variable(tf.zeros(1)),
        'b_ins4': tf.Variable(tf.zeros(1)),

        'full_1': tf.Variable(tf.zeros(1)),
        'full_2': tf.Variable(tf.zeros(1)),
        'full_3': tf.Variable(tf.zeros(1)),
        'full_4': tf.Variable(tf.zeros(1)),
    }

    # Conv 1:
    conv1 = convolution(x, weights['w_conv1'], biases['b_conv1'])

    # Conv 2:
    conv2 = convolution(conv1, weights['w_conv2'], biases['b_conv2'])

    # Conv3:
    conv3 = convolution(conv1, weights['w_conv3'], biases['b_conv3'])

    # Max Pool:
    max_p = maxpool2d(x)

    # Conv 4:
    conv4 = convolution(max_p, weights['w_conv4'], biases['b_conv4'])

    # Flatten
    conv1 = flatten(conv1)
    conv2 = flatten(conv2)
    conv3 = flatten(conv3)
    conv4 = flatten(conv4)

    # Inception
    ins1 = tf.matmul(conv1, weights['w_ins1']) + biases['b_ins1']
    ins2 = tf.matmul(conv2, weights['w_ins2']) + biases['b_ins2']
    ins3 = tf.matmul(conv3, weights['w_ins3']) + biases['b_ins3']
    ins4 = tf.matmul(conv4, weights['w_ins4']) + biases['b_ins4']

    inception = tf.concat(1, [ins1, ins2, ins3, ins4])

    # Fully Connected
    full_1 = tf.matmul(inception, weights['full_1']) + biases['full_1']
    full_1 = tf.nn.relu(full_1)

    full_2 = tf.matmul(full_1, weights['full_2']) + biases['full_2']
    full_2 = tf.nn.relu(full_2)

    full_3 = tf.matmul(full_2, weights['full_3']) + biases['full_3']
    full_3 = tf.nn.relu(full_3)

    full_4 = tf.matmul(full_3, weights['full_4']) + biases['full_4']
    full_4 = tf.nn.relu(full_4)

    logits = full_4

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

    print("Training...")
    print()
    for i in range(EPOCHS):
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x = X_train[offset:end]
            batch_y = y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './lenet')
    print("Model saved")

'''-----------------------------------------------
        Evaluate the Model
-----------------------------------------------'''
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
