import tensorflow as tf
import numpy as np

import sys

class ManyToMany(object):

    def __init__(self, seqlen, num_classes, 
            state_size, batch_size, epochs, 
            learning_rate, 
            ckpt_path,
            model_name='vanilla_net'):

        # attach to object
        self.epochs = epochs
        self.batch_size = batch_size
        self.state_size = state_size
        self.ckpt_path = ckpt_path
        self.model_name = model_name
        self.seqlen = seqlen

        # construct graph
        def __graph__():
            # reset graph
            tf.reset_default_graph()

            # placeholders
            x_ = [ tf.placeholder(tf.int64, [None, ], 
                name = 'x_{}'.format(t)) for t in range(seqlen) ]
            y_ = [ tf.placeholder(tf.int64, [None, ], 
                name = 'y_{}'.format(t)) for t in range(seqlen) ]

            # one-hot
            x_onehot = [ tf.one_hot(x_i_, num_classes) for x_i_ in x_ ]

            # initial state of RNN
            init_state = tf.zeros([batch_size, state_size])

            # rnn cell
            cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
            rnn_outputs, final_state = tf.nn.rnn(cell, x_onehot, init_state)

            # parameters for softmax layer
            W = tf.get_variable('W', [state_size, num_classes])
            b = tf.get_variable('b', [num_classes],
                    initializer=tf.constant_initializer(0.0))

            # output for each time step
            logits = [ tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs ]
            predictions = [ tf.nn.softmax(logit) for logit in logits ]

            # loss
            loss_weights = [ tf.ones([batch_size]) for t in range(seqlen) ]
            losses = tf.nn.seq2seq.sequence_loss_by_example(logits, y_, loss_weights)
            loss = tf.reduce_mean(losses)

            # train op
            train_op = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

            # attach symbols to object, to expose to user of class
            self.x = x_
            self.y = y_
            self.init_state = init_state
            self.train_op = train_op
            self.loss = loss
            self.predictions = predictions

        # run build graph
        sys.stdout.write('<log> Building Graph...')
        __graph__()
        sys.stdout.write('</log>\n')


    def get_feed(self, x, y):
        feed_dict = { x_i_ : x_i for x_i_, x_i in zip(self.x, x) }
        feed_dict.update( { y_i_ : y_i for y_i_, y_i in zip(self.y, y) } )
        feed_dict[self.init_state] = np.zeros([self.batch_size, self.state_size])
        return feed_dict


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
                        feed_dict=self.get_feed(batchX, batchY) )

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
        feed_dict = { self.x[t]: X[t] for t in range(self.seqlen) }
        feed_dict[self.init_state] = np.zeros([self.batch_size, self.state_size])
        predv = sess.run(self.predictions, feed_dict)
        # dec_op_v is a list; also need to transpose 0,1 indices 
        #  (interchange batch_size and timesteps dimensions
        predv = np.array(predv).transpose([1,0,2])
        # return the index of item with highest probability
        return np.argmax(predv, axis=2)


'''

    Sequence to Fixed Length Vector

'''

class ManyToOne(object):

    def __init__(self, seqlen, num_classes, 
            state_size, batch_size, epochs, 
            learning_rate, 
            ckpt_path,
            model_name='many2one'):

        # attach to object
        self.epochs = epochs
        self.batch_size = batch_size
        self.state_size = state_size
        self.ckpt_path = ckpt_path
        self.model_name = model_name
        self.seqlen = seqlen

        # construct graph
        def __graph__():
            # reset graph
            tf.reset_default_graph()

            # placeholders
            x_ = [ tf.placeholder(tf.int64, [None, ], 
                name = 'x_{}'.format(t)) for t in range(seqlen) ]
            y_ = tf.placeholder(tf.int64, [None, ], name = 'y')

            # one-hot
            x_onehot = [ tf.one_hot(x_i_, num_classes) for x_i_ in x_ ]

            # initial state of RNN
            init_state = tf.zeros([self.batch_size, state_size])

            # rnn cell
            cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
            rnn_outputs, final_state = tf.nn.rnn(cell, x_onehot, init_state)

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
            self.init_state = init_state
            self.train_op = train_op
            self.loss = loss
            self.logits = logits

        # run build graph
        sys.stdout.write('<log> Building Graph...')
        __graph__()
        sys.stdout.write('</log>\n')


    def get_feed(self, x, y):
        feed_dict = { x_i_ : x_i for x_i_, x_i in zip(self.x, x) }
        feed_dict.update({ self.y : y })
        feed_dict[self.init_state] = np.zeros([self.batch_size, self.state_size])
        return feed_dict


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
                        feed_dict=self.get_feed(batchX, batchY) )

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
        feed_dict = { self.x[t]: X[t] for t in range(self.seqlen) }
        feed_dict[self.init_state] = np.zeros([self.batch_size, self.state_size])
        predv = sess.run(self.logits, feed_dict)
        # return the index of item with highest probability
        return np.argmax(predv, axis=1)
