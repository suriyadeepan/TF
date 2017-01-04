import tensorflow as tf
import numpy as np

import sys

'''

    Sequence to Fixed Length Vector

'''

class Seq2Vec(object):

    def __init__(self, seqlen, num_classes, 
            state_size, epochs, 
            learning_rate, 
            ckpt_path,
            model_name='seq2vec'):

        # attach to object
        self.epochs = epochs
        self.state_size = state_size
        self.ckpt_path = ckpt_path
        self.model_name = model_name
        self.seqlen = seqlen

        # construct graph
        def __graph__():
            # reset graph
            tf.reset_default_graph()

            # placeholders
            x_ = tf.placeholder(tf.int64, [None, seqlen], name = 'x')
            y_ = tf.placeholder(tf.int64, [None, ], name = 'y')

            # one-hot
            x_onehot = tf.one_hot(x_, num_classes, axis=-1)
            #rnn_inputs = tf.unpack(x_onehot, axis=1)
            # unpack along time axis
            #   http://stackoverflow.com/questions/38728501/inputs-not-a-sequence-wth-rnns-and-tensorflow

            # rnn cell
            cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
            rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=cell, inputs=x_onehot, dtype=tf.float32)

            # rnn_outputs.shape => [ batch_size, seqlen, state_size ]
            #  change to [ seqlen, batch_size, state_size ]
            #   so that rnn_outputs[-1].shape => [ batch_size, state_size ]
            #    note : batch_size -> None
            rnn_outputs = tf.transpose(rnn_outputs, perm=[1,0,2])

            # parameters for softmax layer
            W = tf.get_variable('W', [state_size, num_classes])
            b = tf.get_variable('b', [num_classes],
                    initializer=tf.constant_initializer(0.0))

            # output for each time step
            logits = tf.matmul(rnn_outputs[-1], W) + b
            #predictions = tf.nn.softmax(logit)

            # requires unnormalized prob
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_)
            loss = tf.reduce_mean(losses)

            # train op
            train_op = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

            # attach symbols to object, to expose to user of class
            self.x = x_
            self.y = y_
            self.train_op = train_op
            self.loss = loss
            self.logits = logits

        # run build graph
        sys.stdout.write('<log> Building Graph...')
        __graph__()
        sys.stdout.write('</log>\n')


    def train(self, train_set, sess=None, step=0):

        saver = tf.train.Saver()

        if not sess:
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

        train_loss = 0
        for i in range(step, self.epochs):
            try:
                # get batches
                batchX, batchY = train_set.__next__()
                # run train op
                _, train_loss_ = sess.run([self.train_op, self.loss], 
                        feed_dict= { 
                            self.x : batchX,
                            self.y : batchY
                            })

                # append to losses
                train_loss += train_loss_
                if i and i % 1000 == 0:
                    print('\n>> Average train loss : {}'.format(train_loss/1000))
                    # append avg loss to list
                    train_loss = 0

                    # save model to disk
                    #saver.save(sess, self.ckpt_path + self.model_name + '.ckpt', global_step=i)

            except KeyboardInterrupt:
                print('\n>> Interrupted by user at iteration #' + str(i))
                self.session = sess
                return sess, (i//1000)*1000

        return sess, i


    def restore_last_session(self):
        saver = tf.train.Saver()
        # create a session
        sess = tf.Session()
        # get checkpoint state
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        # restore session
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        # return to user
        return sess


    def predict(self, sess, X):
        predv = sess.run(self.logits, feed_dict = { self.x : X })
        # return the index of item with highest probability
        return np.argmax(predv, axis=1)
