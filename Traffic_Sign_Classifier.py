"""==============================================
        SCRATCH

        Traffic Sign Classifier
==============================================="""


'''-----------------------------------------------
        Import
-----------------------------------------------'''
import pickle
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import cv2

# Clear Graph
tf.reset_default_graph()

# local modules
import helper_functions
'''-----------------------------------------------
        Hyper Parameters
-----------------------------------------------'''
# mu and sigma used for tf.truncated_normal,
# randomly defines variables for the weights and biases for each layer

EPOCHS = 100
BATCH_SIZE = 64

mu = 0
sigma = 0.1

rate = 0.001
epsilon = 1.0

color = cv2.COLOR_RGB2GRAY

'''-----------------------------------------------
        Load Data
-----------------------------------------------'''
# Load Pickled Data
training_file = "images_and_label_data/train.p"
validation_file = "images_and_label_data/valid.p"
testing_file = "images_and_label_data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

train_images, train_labels = train['features'], train['labels']
valid_images, valid_lables = valid['features'], valid['labels']
test_images, test_labels = test['features'], test['labels']


'''-----------------------------------------------
        References to the Data and labels
-----------------------------------------------'''
X_train = np.array(train_images)
y_train = train_labels

X_valid = np.array(valid_images)
y_valid = valid_lables

X_test = test_images
y_test = test_labels

'''-----------------------------------------------
        Visualize the Data
-----------------------------------------------'''

n_train = len(X_train)
n_validation = len(X_valid)
n_test = len(X_test)
image_shape = X_train[0].shape
n_classes = len(np.unique(y_train))

print("X Train length ", len(X_train))
logs_path = "./tensor_board"

'''=============================================
        Preprocess
=============================================='''

'''--------------------------
        Balance Data-set
-----------------------------'''

# Merge training and validation images
X_train = np.concatenate((X_train, X_valid), axis=0)
y_train = np.concatenate((y_train, y_valid), axis=0)

hist, bins = np.histogram(y_train, bins=n_classes)
average = 2000

image_copies = []
label_copies = []


for image, label in zip(X_train, y_train):
    if hist[label] < average:
        difference = (average - hist[label])
        x_range = int(difference / hist[label])
        for i in range(x_range):
            new_image = helper_functions.random_translate(image)
            new_image = helper_functions.random_brightness(image)
            new_image = helper_functions.random_scale(image)
            image_copies.append(new_image)
            label_copies.append(label)

X_train = np.concatenate((X_train, image_copies), axis=0)
y_train = np.concatenate((y_train, label_copies), axis=0)

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.20, random_state=42, shuffle=True)

'''--------------------------
        color
-----------------------------'''
if color is not None:
    X_train_new = []
    X_test_new = []
    X_valid_new = []

    print("Color = ", color)
    for image in X_train:
        new_img = cv2.cvtColor(image, color)
        X_train_new.append(new_img)

    for image in X_test:
        new_img = cv2.cvtColor(image, color)
        X_test_new.append(new_img)

    for image in X_valid:
        new_img = cv2.cvtColor(image, color)
        X_valid_new.append(new_img)

    X_train_new = np.array(X_train_new)
    X_test_new = np.array(X_test_new)
    X_valid_new = np.array(X_valid_new)

    X_train = X_train_new
    X_test = X_test_new
    X_valid = X_valid_new

    # Show original and color changed image
    print("Color Changed: ", X_train_new.shape)


'''--------------------------
        Shuffle
-----------------------------'''
# Shuffle
X_train, y_train = shuffle(X_train, y_train)
X_valid, y_valid = shuffle(X_valid, y_valid)
X_test, y_test = shuffle(X_test, y_test)


'''--------------------------
        Normalize
-----------------------------'''
img_max = np.max(X_train)

X_train = (X_train / img_max) - 0.5
X_valid = (X_valid / img_max) - 0.5
X_test = (X_test / img_max) - 0.5

print("Mean 2", np.mean(X_train))


'''============================================
            LeNet
==============================================='''


def convolution(x, W, b):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
    conv = tf.nn.bias_add(conv, b)
    conv = tf.nn.elu(conv)
    return conv


def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


def LeNet(x):

    # Conv 1:
    conv1 = convolution(x, weights['w_conv1'], biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    # Conv 2
    conv2 = convolution(conv1, weights['w_conv2'], biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    # Conv 3
    conv3 = convolution(conv2, weights['w_conv3'], biases['b_conv3'])

    # Flatten
    conv2_flat = flatten(conv2)
    conv3_flat = flatten(conv3)
    fc0 = tf.concat([conv2_flat, conv3_flat], 1)

    # Dropout
    fc0 = tf.nn.dropout(fc0, dropout)

    # Fully Connected 1
    fc1 = tf.matmul(fc0, weights['w_fc1']) + biases['b_fc1']
    fc1 = tf.nn.elu(fc1)

    # Fully Connected 2
    fc2 = tf.matmul(fc1, weights['w_fc2']) + biases['b_fc2']
    fc2 = tf.nn.elu(fc2)

    # Fully Connected 3
    logits = tf.matmul(fc2, weights['w_fc3']) + biases['b_fc3']

    return logits
'''============================================'''


'''-----------------------------------------------
        Features and Labels
-----------------------------------------------'''

'''--------------------------
        Input
-----------------------------'''

# Get Image depth (color channels)
if len(X_train.shape) > 3:
    placeholder_shape = (None, 32, 32, 3)
    depth = 3
else:
    placeholder_shape = (None, 32, 32, 1)
    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)
    X_valid = np.expand_dims(X_valid, axis=3)
    depth = 1

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, placeholder_shape, name='x-input')
    y = tf.placeholder(tf.int32, None)
    one_hot_y = tf.one_hot(y, n_classes, name='y-input')

# Dropout Rate
dropout = tf.placeholder_with_default(0.5, shape=())


'''--------------------------
        Weights
-----------------------------'''
with tf.name_scope('weights'):
    # Weights
    weights = {
        'w_conv1':  tf.Variable(tf.truncated_normal([5, 5, depth, 6], mean=mu, stddev=sigma)),
        'w_conv2':  tf.Variable(tf.truncated_normal([5, 5, 6, 16], mean=mu, stddev=sigma)),
        'w_conv3':  tf.Variable(tf.truncated_normal([5, 5, 16, 400], mean=mu, stddev=sigma)),
        'w_fc1': tf.Variable(tf.truncated_normal([800, 120], mean=mu, stddev=sigma)),
        'w_fc2': tf.Variable(tf.truncated_normal([120, 84], mean=mu, stddev=sigma)),
        'w_fc3': tf.Variable(tf.truncated_normal([84, 43], mean=mu, stddev=sigma)),
    }



'''--------------------------
        Biases
-----------------------------'''
with tf.name_scope('biases'):
    # Biases
    biases = {
        'b_conv1':  tf.Variable(tf.zeros(6)),
        'b_conv2':  tf.Variable(tf.zeros(16)),
        'b_conv3':  tf.Variable(tf.zeros(400)),
        'b_fc1': tf.Variable(tf.zeros(120)),
        'b_fc2': tf.Variable(tf.zeros(84)),
        'b_fc3': tf.Variable(tf.zeros(43))

    }


'''-----------------------------------------------
        Training Pipeline
-----------------------------------------------'''
logits = LeNet(x)

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)

with tf.name_scope('los_operation'):
    loss_operation = tf.reduce_mean(cross_entropy)

with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer()

with tf.name_scope('training_operation'):
    training_operation = optimizer.minimize(loss_operation)

with tf.name_scope('Accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    with tf.name_scope('accuracy'):
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Create a summary from cost and accuracy
tf.summary.scalar("cost", loss_operation)
tf.summary.scalar("accuracy", accuracy_operation)

# Merge all summaries
merged = tf.summary.merge_all()

saver = tf.train.Saver()

'''-----------------------------------------------
        Model Evaluation
-----------------------------------------------'''




def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, dropout: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

'''-----------------------------------------------
        Train the Model
-----------------------------------------------'''
with tf.Session() as sess:
    i=0
    training=True

    # Initialize variables
    sess.run(tf.global_variables_initializer())

    # Create log writer object
    train_writer = tf.summary.FileWriter(logs_path + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(logs_path + '/test', tf.global_variables_initializer().run())

    # Sanity Checks
    print("\n\n Sanity Checks\n-------------------------------------------------\n")
    num_examples = len(X_train)
    print(":: Original :: ", train_images.shape)
    print("X Train :: length = ", len(X_train), "shape = ", X_train.shape)
    print("X Valid :: length = ", len(X_valid), "shape = ", X_valid.shape)
    print("X Test :: length = ", len(X_test), "shape = ", X_test.shape)
    print("\n-------------------------------------------------\nTraining\n\n\n")
    print("===============================")
    print("Epoch #  || Validation Accuracy")
    print("===============================")

    for epoch in range(EPOCHS):

        # batches in one epoch
        batch_count = int(num_examples / BATCH_SIZE)
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            _, summary = sess.run([training_operation, merged], feed_dict={x: batch_x, y: batch_y})

            # Write Test Log
            train_writer.add_summary(summary, i)
            i += 1

        # Write Train Log
        summary, acc = sess.run([merged, accuracy_operation], feed_dict={x: X_valid, y: y_valid})
        test_writer.add_summary(summary, i * epoch)

        validation_accuracy = evaluate(X_valid, y_valid)
        accuracy_percent = validation_accuracy * 100.0
        print("EPOCH {:2d} || {:.2f} %".format((epoch + 1), accuracy_percent))
        print()
        if validation_accuracy >= 0.997:
            break

    # Close Writers
    train_writer.close()
    test_writer.close()

    # Save Model
    saver.save(sess, './lenet')
    print("Model saved")
    print("\n\n---------------------------------------\n")
    print("----------------------------")
    print("Testing\n----------------------------")

'''-----------------------------------------------
        Evaluate the Model
-----------------------------------------------'''
with tf.Session() as sess:
    saver.restore(sess, './lenet')

    test_accuracy = evaluate(X_test, y_test)
    accuracy_percent = test_accuracy * 100.0
    print("Test Accuracy = {:.2f}".format(accuracy_percent))
