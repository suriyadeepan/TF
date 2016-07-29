#cp -r ../mini/input_data.py ../mini/MNIST_data .

import tensorflow as tf
import numpy as np
import input_data

import math

# dataset
mnist = input_data.read_data_sets('MNIST_data/')

# Start building the graph
#   INPUT placeholders
x = tf.placeholder(tf.float32,[None,784],name = 'X')
y = tf.placeholder(tf.int32,[None],name = 'Y')


# model
h1 = tf.nn.relu( tf.matmul(x,tf.Variable(tf.truncated_normal([784,500],stddev=1.0 / math.sqrt(float(784))))) + tf.Variable(tf.zeros(500)) )
h2 = tf.nn.relu( tf.matmul(h1,tf.Variable(tf.truncated_normal([500,500],stddev=1.0 / math.sqrt(float(784))))) + tf.Variable(tf.zeros(500)) )
_y = tf.matmul(h2,tf.Variable(tf.truncated_normal([500,10],stddev=1.0 / math.sqrt(float(784))))) + tf.Variable(tf.zeros(10)) 

# loss
labels = tf.to_int64(y)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(_y, labels, name='xentropy')
loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')


# train op
optimizer = tf.train.GradientDescentOptimizer(0.1)
global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = optimizer.minimize(loss, global_step=global_step)


# start session, init variables
sess = tf.Session()
sess.run(tf.initialize_all_variables())


# training
for i in range(10000):
    batchX, batchY = mnist.train.next_batch(128)
    _, loss_val = sess.run([train_op,loss], feed_dict={x : batchX,y : batchY})

# Evaluate
correct = tf.nn.in_top_k(_y, y, 1)
eval_op = tf.reduce_mean(tf.cast(correct, tf.float32))

testX, testY = mnist.test.next_batch(128)
print 'Accuracy :', sess.run(eval_op,feed_dict={x:testX, y:testY})
