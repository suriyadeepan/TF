import tensorflow as tf
import numpy as np
import input_data


# get MNIST data
mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

trainX,trainY = mnist.train.next_batch(100)

# _y = Wx + b
# Model
x = tf.placeholder(tf.float32)
W = tf.Variable(tf.zeros([784,10]),name='weight')
b = tf.Variable(tf.zeros(10),name='bias')
_y = tf.nn.softmax(tf.matmul(x,W) + b)

y = tf.placeholder(tf.float32,[None,10])

# cost : squared error? (y-_y)^2
cce = tf.nn.softmax_cross_entropy_with_logits(_y,y)

train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cce)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(2000):
    batchX, batchY = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x : batchX,y : batchY})

# Accuracy
eq = tf.equal(tf.argmax(y,1),tf.argmax(_y,1))
accuracy = tf.reduce_mean(tf.cast(eq,tf.float32))


testX, testY = mnist.test.next_batch(100)
acc_val = sess.run(accuracy,feed_dict= {x : testX, y : testY})

print '\n>>Accuracy :', acc_val
