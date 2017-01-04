import data
import model

if __name__ == '__main__':
    # generate dataset
    X, Y = data.gen_data()

    # hyper-parameters
    seqlen = X.shape[-1]
    num_classes = 20 # binary
    state_size = 16  # num of units in rnn's internal state
    batch_size = 128
    epochs = 100000 # 100_000 (need this <- python 3.6)
    learning_rate = 0.1


    # build model
    net = model.ManyToOne(
            seqlen = seqlen,
            num_classes = num_classes,
            state_size = state_size,
            batch_size = batch_size,
            epochs = epochs,
            learning_rate = learning_rate,
            ckpt_path='ckpt/'
            )


    # build batch generator for training
    train_set = data.rand_batch_gen(X, Y, batch_size)

    # train model
    sess, last_step, train_losses = net.train(train_set)

    print('\n>> After training')
    print('\t:: Train Losses : \n{}'.format(train_losses))
