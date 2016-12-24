import tensorflow as tf
import numpy as np 
import data, data_utils

import sys

# gather dataset
data_ctl, idx_words, idx_phonemes = data.load_data()
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_words, idx_phonemes)


# parameters 
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 128
xvocab_size = len(data_ctl['idx2alpha'].keys())  # 27
yvocab_size = len(data_ctl['idx2pho'].keys())  # 70
emb_dim = 128


'''
 build the graph

'''
tf.reset_default_graph()

enc_ip = [ tf.placeholder(dtype=tf.int32,
                       shape = (None,),
                       name = 'ei_{}'.format(i)) for i in range(xseq_len) ]
# alternatively
#  enc_ip = tf.placeholder(shape=[None,xseq_len], dtype=tf.int32, name='enc_ip')
labels = [ tf.placeholder(dtype=tf.int32,
                       shape = (None,),
                       name = 'ei_{}'.format(i)) for i in range(yseq_len) ]
# alternatively
#  labels = tf.placeholder(shape=[None,yseq_len], dtype=tf.int32, name='labels')
dec_ip = [ tf.zeros_like(enc_ip[0], dtype=tf.int32, name='GO')] + labels[:-1]

keep_prob = tf.placeholder(tf.float32)
basic_cell = tf.nn.rnn_cell.DropoutWrapper(
        tf.nn.rnn_cell.BasicLSTMCell(emb_dim, state_is_tuple=True),
        output_keep_prob=keep_prob)
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([basic_cell]*3, state_is_tuple=True)


with tf.variable_scope('decoder') as scope:
    decode_outputs, decode_states = tf.nn.seq2seq.embedding_rnn_seq2seq(enc_ip,dec_ip, stacked_lstm,
                                        xvocab_size, yvocab_size, emb_dim)
    scope.reuse_variables()
    # testing
    decode_outputs_test, decode_states_test = tf.nn.seq2seq.embedding_rnn_seq2seq(
        enc_ip, dec_ip, stacked_lstm, xvocab_size, yvocab_size,emb_dim,
        feed_previous=True)

# we weight the losses based on timestep of decoder output
loss_weights = [tf.ones_like(l, dtype=tf.float32) for l in labels] # gives [1, 1, ..., 1,1] - equal weights
loss = tf.nn.seq2seq.sequence_loss(decode_outputs, labels, loss_weights, yvocab_size)
train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)


'''
 training and evaluation functions

'''
def get_feed(X, Y):
    feed_dict = {enc_ip[t]: X[t] for t in range(xseq_len)}
    feed_dict.update({labels[t]: Y[t] for t in range(yseq_len)})
    return feed_dict

def train_batch(train_batch_gen):
    # get batches
    batchX, batchY = train_batch_gen.__next__()
    # build feed
    feed_dict = get_feed(batchX, batchY)
    feed_dict[keep_prob] = 0.5
    _, loss_v = sess.run([train_op, loss], feed_dict)
    return loss_v

def eval_step(eval_batch_gen):
    # get batches
    batchX, batchY = eval_batch_gen.__next__()
    # build feed
    feed_dict = get_feed(batchX, batchY)
    feed_dict[keep_prob] = 1.
    loss_v, dec_op_v = sess.run([loss, decode_outputs_test], feed_dict)
    # dec_op_v is a list; also need to transpose 0,1 indices
    dec_op_v = np.array(dec_op_v).transpose([1,0,2])
    return loss_v, dec_op_v, batchX, batchY

def eval_batch(eval_batch_gen, num_batches):
    losses, predict_loss = [], []
    for i in range(num_batches):
        loss_v, dec_op_v, batchX, batchY = eval_step(eval_batch_gen)
        losses.append(loss_v)
        for j in range(len(dec_op_v)):
            real = batchX.T[j]
            predict = np.argmax(dec_op_v, axis=2)[j]
            predict_loss.append(all(real == predict))
    return np.mean(losses), np.mean(predict_loss)

if __name__ == '__main__':
    val_batch_gen = data_utils.rand_batch_gen(validX, validY, 16)
    train_eval_batch_gen = data_utils.rand_batch_gen(trainX, trainY, 16)
    train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, 128)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for i in range(100000):
            try:
                train_batch(train_batch_gen)
                if i % 1000 == 0:
                    val_loss, val_predict = eval_batch(val_batch_gen, 16)
                    train_loss, train_predict = eval_batch(train_eval_batch_gen, 16)
                    print("val loss   : {0}, val predict   = {1}%".format(val_loss, val_predict * 100))
                    print("train loss : {0}, train predict = {1}%".format(train_loss, train_predict * 100))
                    print
                    sys.stdout.flush()
            except KeyboardInterrupt:
                print("interrupted by user")
                break










