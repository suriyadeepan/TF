import tensorflow as tf
import numpy as np

import sys

class VanillaNet(object):

    def __init__(self, timesteps, num_classes, 
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
            #y_as_list = tf.unpack(y, axis=1)
            y_as_list = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(1, timesteps, y)]
            # loss
            loss_weights = [ tf.ones([batch_size]) for i in range(timesteps) ]
            losses = tf.nn.seq2seq.sequence_loss_by_example(logits, y_as_list, loss_weights)
            avg_loss = tf.reduce_mean(losses)

            # train op
            train_op = tf.train.AdagradOptimizer(learning_rate).minimize(avg_loss)

            # attach symbols to object, to expose to user of class
            self.x = x
            self.y = y
            self.init_state = init_state
            self.train_op = train_op
            self.loss = avg_loss
            self.predictions = predictions

        # run build graph
        sys.stdout.write('<log> Building Graph...')
        __graph__()
        sys.stdout.write('</log>\n')

    def train(self, train_set, sess=None, step=0):

        saver = tf.train.Saver()

        if not sess:
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

        train_losses = []
        train_loss = 0
        for i in range(step, self.epochs):
            try:
                train_init_state = np.zeros([self.batch_size, self.state_size])
                # get batches
                batchX, batchY = train_set.__next__()
                # run train op
                _, train_loss_, pred_ = sess.run([self.train_op, self.loss, self.predictions], \
                        feed_dict = { self.x : batchX, 
                            self.y : batchY,
                            self.init_state : train_init_state })

                #print('predicted : {}'.format(np.argmax(pred_[0], axis=1)))
                #print('actual    : {}'.format(batchY))
                    
                # append to losses
                train_loss += train_loss_
                if i and i % 1000 == 0:
                    print('\n>> Average train loss : {}'.format(train_loss/1000))
                    # append avg loss to list
                    train_losses.append(train_loss/1000)
                    train_loss = 0

                    # save model to disk
                    #saver.save(sess, self.ckpt_path + self.model_name + '.ckpt', global_step=i)
            except KeyboardInterrupt:
                print('\n>> Interrupted by user at iteration #' + str(i))
                self.session = sess
                return sess, (i//1000)*1000, train_losses

        return sess, i, train_losses


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


