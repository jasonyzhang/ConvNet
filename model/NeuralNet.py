import tensorflow as tf
import numpy as np
from scipy.io import loadmat
from sklearn import utils

class NeuralNet(object):
    """A convolutional neural network.
    
    :param str save_file: Name of save file.
    :param float lamb: Lambda for l2 regularization.
    :param float epsilon: Epsilon for batch normalization.
    :param bool use_augmented: If True, uses training data augmented with transformations.
    :param fn pre_fn: Preprocessing function for training data. Default is standardization.
    """

    def __init__(self, save_file, lamb=0., epsilon=1e-3, use_augmented=False, pre_fn=None):

        self.save_file = save_file
        
        # Hyper-parameters

        # Fully connected layers size
        self.FC_DIM_1 = 1024
        self.FC_DIM_2 = 512

        # L2 regularization
        self.lamb = lamb

        # Batch normalization
        self.epsilon = epsilon

        self.use_augmented = use_augmented

        if pre_fn:
            self.pre_fn = pre_fn
        else:
            self.pre_fn = lambda x: ((x - np.mean(x)) / np.std(x))

    def build_graph(self, training=True):
        """Builds the structure of the CNN.
        
        :param bool training: If True, graph is built to train.
        :return inputs: Placeholder for input images of dimension Nx784.
        :return labels: Placeholder for image labels of dimension Nx26.
        :return keep: Placeholder for keep probability for dropout.
        :return train: Training operation.
        :return loss: Loss operation.
        :return logits: Logit operation.
        :return Saver: Saver object.
        :rtype: (Placeholder, Placeholder, Placeholder, Operation, Operation, Operation, Saver)
        """

        # Variables
        inputs = tf.placeholder(tf.float32,
                                     shape=(None, 784),
                                     name='input')
        labels = tf.placeholder(tf.float32,
                                     shape=[None, 26],
                                     name='labels')
        keep = tf.placeholder(tf.float32, name='keep')

        W_conv_1 = tf.Variable(tf.truncated_normal(
            shape=[5, 5, 1, 32],
            mean=0.,
            stddev=0.1,
        ))
        b_conv_1 = tf.Variable(np.zeros(32).astype(np.float32))

        W_conv_2 = tf.Variable(tf.truncated_normal(
            shape=[5, 5, 32, 64],
            mean=0.,
            stddev=0.1,
        ))

        b_conv_2 = tf.Variable(np.zeros(64).astype(np.float32))

        W_fc_1 = tf.Variable(tf.truncated_normal(
            shape=[7 * 7 * 64, self.FC_DIM_1],
            mean=0.,
            stddev=1./8.
        ))

        b_fc_1 = tf.Variable(np.zeros(self.FC_DIM_1).astype(np.float32))

        W_fc_2 = tf.Variable(tf.truncated_normal(
            shape=[self.FC_DIM_1, self.FC_DIM_2],
            mean=0.,
            stddev=1./np.sqrt(self.FC_DIM_1)
        ))

        b_fc_2 = tf.Variable(np.zeros(self.FC_DIM_2).astype(np.float32))

        W_fc_3 = tf.Variable(tf.truncated_normal(
            shape=[self.FC_DIM_2, 26],
            mean=0.,
            stddev=1./np.sqrt(self.FC_DIM_2)
        ))

        b_fc_3 = tf.Variable(np.zeros(26).astype(np.float32))

        # bn_input = self.batch_normalization(inputs, training)
        input_layer = tf.reshape(inputs, [-1, 28, 28, 1])

        z_conv_1 = tf.nn.conv2d(
            input=input_layer,
            filter=W_conv_1,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )
        # bn_conv_1 = self.batch_normalization(z_conv_1, training)
        h_conv_1 = tf.nn.relu(z_conv_1 + b_conv_1)

        h_pool_1 = tf.nn.max_pool(
            value=h_conv_1,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME'
        )

        z_conv_2 = tf.nn.conv2d(
            input=h_pool_1,
            filter=W_conv_2,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )
        # bn_conv_2 = self.batch_normalization(z_conv_2, training)
        h_conv_2 = tf.nn.relu(z_conv_2 + b_conv_2)

        h_pool_2 = tf.reshape(tf.nn.max_pool(
            value=h_conv_2,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME'
        ), [-1, 7 * 7 * 64])

        z_fc_1 = tf.matmul(h_pool_2, W_fc_1)
        bn_fc_1 = self.batch_normalization(z_fc_1, training)
        h_fc_1 = tf.nn.relu(bn_fc_1 + b_fc_1)
        h_drop_1 = tf.nn.dropout(h_fc_1, keep)

        z_fc_2 = tf.matmul(h_drop_1, W_fc_2)
        bn_fc_2 = self.batch_normalization(z_fc_2, training)

        h_fc_2 = tf.nn.relu(bn_fc_2 + b_fc_2)
        h_drop_2 = tf.nn.dropout(h_fc_2, keep)

        logits = tf.matmul(h_drop_2, W_fc_3) + b_fc_3
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=labels
        ))

        regularizers = (
            tf.nn.l2_loss(W_conv_1) + tf.nn.l2_loss(b_conv_1) +
            tf.nn.l2_loss(W_conv_2) + tf.nn.l2_loss(b_conv_2) +
            tf.nn.l2_loss(W_fc_1) + tf.nn.l2_loss(b_fc_1) +
            tf.nn.l2_loss(W_fc_2) + tf.nn.l2_loss(b_fc_2) +
            tf.nn.l2_loss(W_fc_3) + tf.nn.l2_loss(b_fc_3)
        )

        new_loss = loss + self.lamb * regularizers

        train = tf.train.AdamOptimizer(1e-4).minimize(new_loss)

        return inputs, labels, keep, train, loss, logits, tf.train.Saver()

    def batch_normalization(self, inputs, training, decay=0.999):
        """Wrapper for creating a batch normalization layer.
        
        :param tensor inputs: Input layer.
        :param bool training: If True, runs train operations.
        :param float decay: Decay rate for exponential decay.
        :return: Batch normalization layer.
        """

        # Shape and scale
        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))

        mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]),
                               trainable=False)
        var = tf.Variable(tf.ones([inputs.get_shape()[-1]]),
                              trainable=False)

        if training:
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
            train_mean = tf.assign(mean,
                                   mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(var, var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs,
                                                 batch_mean, batch_var, beta,
                                                 scale, self.epsilon)
        else:
            return tf.nn.batch_normalization(inputs, mean, var, beta,
                                             scale, self.epsilon)

    def get_data(self):
        """Retrieves saved data for training and validation.
        
        :return: Batch generator, Validation images, Validation labels
        """
        shuffle = np.load('data/data.npy')
        data = loadmat('data/letters_data.mat')
        train_x = self.pre_fn(data['train_x'])



        # y needs to be one hot encoded
        old_y = data['train_y'] - 1
        train_y = np.zeros((len(old_y), 26))
        for i in range(len(old_y)):
            train_y[i, old_y[i]] = 1
        tx = train_x[shuffle][:100000]
        ty = train_y[shuffle][:100000]

        if self.use_augmented:
            print 'Using augmented data'
            training_data = loadmat('data/augmented_data_compressed.mat')
            tx = self.pre_fn(training_data['tx'])
            ty = training_data['ty']
            print 'done loading'

        vx = train_x[shuffle][100000:]
        vy = train_y[shuffle][100000:]

        def batch_generator(n):
            while True:
                _tx, _ty = utils.shuffle(tx, ty)
                for i in xrange(len(tx)/n):
                    yield _tx[i * n: (i + 1) * n], _ty[i * n: (i + 1) * n]

        return batch_generator, vx, vy

    def train(self, num_epochs=2., batch_size=100, keep_prob=1., resume=True, log=True):

        tf.reset_default_graph()

        inputs, labels, keep, train, loss, logits, saver = self.build_graph(True)

        batch_generator, vx, vy = self.get_data()

        batch_gen = batch_generator(batch_size)

        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)

            if resume:
                saver.restore(session, self.save_file)

            validation_err = 1

            for i in xrange(int(100000. * num_epochs / batch_size)):
                batch_x, batch_y = next(batch_gen)

                if i%100 == 0:
                    train_pred = session.run(logits, {
                        inputs: batch_x,
                        labels: batch_y,
                        keep: 1.
                    })

                    train_err = error(np.argmax(train_pred, 1),
                                      np.argmax(batch_y, 1))
                    if log:
                        print i, 1 - train_err

                train.run({
                    inputs: batch_x,
                    labels: batch_y,
                    keep: keep_prob
                })

                # self.saver.save(session, 'model-1')
                if i % 1000 == 999:
                    validation_pred = session.run(logits, {
                        inputs: vx,
                        labels: vy,
                        keep: 1.
                    })
                    validation_err = error(np.argmax(validation_pred, 1),
                                           np.argmax(vy, 1))
                    print 'validation acc:', 1 - validation_err
                    saver.save(session, self.save_file)

            saver.save(session, self.save_file)
            return 1 - validation_err

    def predict(self, data):
        tf.reset_default_graph()
        inputs, labels, keep, train, loss, logits, saver = self.build_graph(
            False)


        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            saver.restore(session, self.save_file)
            predictions = session.run(logits, {
                inputs: data,
                keep: 1,
                labels: np.array([[0] * 26] * len(data[0]))
            })
        return predictions


def error(p, t):
    return np.mean(p != t)
