import tensorflow as tf

import sys

class VanillaNet(object):

    def __init__(self, timesteps, num_classes, state_size, batch_size, epochs, learning_rate, model_name='vanilla_net'):

        # construct graph
        def __graph__():
            # reset graph
            tf.reset_default_graph()

            # placeholders
            x = tf.placeholder(tf.int32, [batch_size, timesteps], name = 'x')
            y = tf.placeholder(tf.int32, [batch_size, timesteps], name = 'x')
            # one-hot
            x_onehot = tf.one_hot(x, num_classes) # [batch_size x timesteps x num_classes]
            rnn_inputs = tf.unstack(x_onehot, axis=1) # [batch_size x num_classes]
            # initial state of RNN
            init_state = tf.zeros([batch_size, state_size])

            # rnn cell
            cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
            rnn_outputs, final_state = tf.nn.rnn(cell, rnn_inputs, init_state)

            # parameters for softmax layer
            W = tf.get_variable('W', [state_size, num_classes])
            b = tf.get_variable('b', [num_classes],
                    initializer=tf.constant_initializer(0.0))

            # output for each time step
            logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
            predictions = [tf.nn.softmax(logit) for logit in logits]

            # unpack y
            y_as_list = tf.unpack(y, axis=1)
            # loss
            loss_weights = [ tf.zeros([batch_size]) for i in range(timesteps) ]
            losses = tf.nn.seq2seq.sequence_loss_by_example(logits, y_as_list, loss_weights)
            avg_loss = tf.reduce_mean(losses)

            # train op
            train_op = tf.train.AdagradOptimizer(learning_rate).minimize(avg_loss)

            # attach symbols to object, to expose to user of class
            self.x = x
            self.y = y
            self.train_op = train_op
            self.loss = avg_loss
            self.predictions = predictions

        # run build graph
        sys.stdout.write('<log> Building Graph...')
        __graph__()
        sys.stdout.write('</log>\n')

    def train(self, sess=None):

        sess = tf.Session() if not sess else sess












