import tensorflow as tf
import numpy as np
import matplotlib
from scipy.io import loadmat


class NeuralNet(object):

    def __init__(self, save_file, lamb):

        self.save_file = save_file
        self.lamb = lamb

    def initialize(self):
        # Hyper-parameters

        # Fully connected layers size
        FC_DIM_1 = 1024
        FC_DIM_2 = 512

        # batch normalization
        epsilon = 1e-3

        # Variables
        self.inputs = tf.placeholder(tf.float32,
                                     shape=(None, 784),
                                     name='input')
        self.labels = tf.placeholder(tf.float32,
                                     shape=[None, 26],
                                     name='labels')
        self.keep = tf.placeholder(tf.float32, name='keep')

        self.W_conv_1 = tf.Variable(tf.truncated_normal(
            shape=[5, 5, 1, 32],
            mean=0.,
            stddev=0.1,
        ))
        self.b_conv_1 = tf.Variable(np.zeros(32).astype(np.float32))

        self.W_conv_2 = tf.Variable(tf.truncated_normal(
            shape=[5, 5, 32, 64],
            mean=0.,
            stddev=0.1,
        ))

        self.b_conv_2 = tf.Variable(np.zeros(64).astype(np.float32))

        self.W_fc_1 = tf.Variable(tf.truncated_normal(
            shape=[7 * 7 * 64, FC_DIM_1],
            mean=0.,
            stddev=1./8.
        ))

        self.b_fc_1 = tf.Variable(np.zeros(FC_DIM_1).astype(np.float32))

        self.W_fc_2 = tf.Variable(tf.truncated_normal(
            shape=[FC_DIM_1, FC_DIM_2],
            mean=0.,
            stddev=1./np.sqrt(FC_DIM_1)
        ))

        self.b_fc_2 = tf.Variable(np.zeros(FC_DIM_2).astype(np.float32))

        self.W_fc_3 = tf.Variable(tf.truncated_normal(
            shape=[FC_DIM_2, 26],
            mean=0.,
            stddev=1./np.sqrt(FC_DIM_2)
        ))

        self.b_fc_3 = tf.Variable(np.zeros(26).astype(np.float32))

        # Scale and shift for batch normalization
        self.scale1 = tf.Variable(tf.ones([FC_DIM_1]))
        self.beta1 = tf.Variable(tf.zeros([FC_DIM_1]))
        self.scale2 = tf.Variable(tf.ones([FC_DIM_2]))
        self.beta2 = tf.Variable(tf.zeros([FC_DIM_2]))


        self.input_layer = tf.reshape(self.inputs, [-1, 28, 28, 1])

        h_conv_1 = tf.nn.relu(tf.nn.conv2d(
            input=self.input_layer,
            filter=self.W_conv_1,
            strides=[1, 1, 1, 1],
            padding='SAME'
        ) + self.b_conv_1)

        h_pool_1 = tf.nn.max_pool(
            value=h_conv_1,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME'
        )

        h_conv_2 = tf.nn.relu(tf.nn.conv2d(
            input=h_pool_1,
            filter=self.W_conv_2,
            strides=[1, 1, 1, 1],
            padding='SAME'
        ) + self.b_conv_2)

        h_pool_2 = tf.reshape(tf.nn.max_pool(
            value=h_conv_2,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME'
        ), [-1, 7 * 7 * 64])

        z_bn_1 = tf.matmul(h_pool_2, self.W_fc_1)
        batch_mean_1, batch_var_1 = tf.nn.moments(z_bn_1, [0])
        h_bn_1 = tf.nn.batch_normalization(
            x=z_bn_1,
            mean=batch_mean_1,
            variance=batch_var_1,
            offset=self.beta1,
            scale=self.scale1,
            variance_epsilon=epsilon
        )
        h_fc_1 = tf.nn.relu(h_bn_1 + self.b_fc_1)
        h_drop_1 = tf.nn.dropout(h_fc_1, self.keep)

        z_bn_2 = tf.matmul(h_drop_1, self.W_fc_2)
        batch_mean_2, batch_var_2 = tf.nn.moments(z_bn_2, [0])
        h_bn_2 = tf.nn.batch_normalization(
            x=z_bn_2,
            mean=batch_mean_2,
            variance=batch_var_2,
            offset=self.beta2,
            scale=self.scale2,
            variance_epsilon=epsilon
        )

        h_fc_2 = tf.nn.relu(h_bn_2 + self.b_fc_2)
        h_drop_2 = tf.nn.dropout(h_fc_2, self.keep)

        logits = tf.matmul(h_drop_2, self.W_fc_3) + self.b_fc_3
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self.labels
        ))

        self.saver = tf.train.Saver()
        self.predict = logits
        return cost

    def get_data(self):
        shuffle = np.load('data.npy')
        data = loadmat("letters_data.mat")
        train_x = data['train_x']

        # y needs to be one hot encoded
        old_y = data['train_y'] - 1
        train_y = np.zeros((len(old_y), 26))
        for i in range(len(old_y)):
            train_y[i, old_y[i]] = 1
        tx = train_x[shuffle][:100000]
        ty = train_y[shuffle][:100000]
        vx = train_x[shuffle][100000:]
        vy = train_y[shuffle][100000:]

        def batch_generator(n):
            while True:
                s = np.random.choice(np.arange(100000), replace=False,
                                           size=100000)
                for i in range(100000/n):
                    yield tx[s][i * n: (i + 1) * n], ty[s][i * n: (i + 1) * n]

        return batch_generator, vx, vy

    def train(self, num_epochs=2., batch_size=100, keep_prob=1.):

        loss = self.initialize()
        """
        regularizers = (
            tf.nn.l2_loss(self.W_conv_1) + tf.nn.l2_loss(self.b_conv_1) +
            tf.nn.l2_loss(self.W_conv_2) + tf.nn.l2_loss(self.b_conv_2) +
            tf.nn.l2_loss(self.W_fc_1) + tf.nn.l2_loss(self.b_fc_1) +
            tf.nn.l2_loss(self.W_fc_2) + tf.nn.l2_loss(self.b_fc_2) +
            tf.nn.l2_loss(self.W_fc_3) + tf.nn.l2_loss(self.b_fc_3)
        )"""
        regularizers = 0

        new_loss = loss + self.lamb * regularizers


        #train = tf.train.MomentumOptimizer(10e-4, 0.9).minimize(loss)
        train = tf.train.AdamOptimizer(1e-4).minimize(new_loss)

        batch_generator, vx, vy = self.get_data()

        batch_gen = batch_generator(batch_size)

        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)

            for i in xrange(int(100000. * num_epochs / batch_size)):
                batch_x, batch_y = next(batch_gen)

                if i%100 == 0:
                    train_pred = session.run(self.predict, {
                        self.inputs: batch_x,
                        self.labels: batch_y,
                        self.keep: 1.
                    })
                    train_err = error(np.argmax(train_pred, 1),
                                      np.argmax(batch_y, 1))
                    print i, 1 - train_err

                train.run({
                    self.inputs: batch_x,
                    self.labels: batch_y,
                    self.keep: keep_prob
                })

                #self.saver.save(session, 'model-1')
                if i % 1000 == 999:
                    validation_pred = session.run(self.predict, {
                        self.inputs: vx,
                        self.labels: vy,
                        self.keep: 1.
                    })
                    validation_err = error(np.argmax(validation_pred, 1),
                                           np.argmax(vy, 1))
                    print 'validation acc:', 1 - validation_err




def error(p, t):
    return np.mean(p != t)
