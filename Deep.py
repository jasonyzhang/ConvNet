
# coding: utf-8

# In[1]:


# https://www.tensorflow.org/tutorials/layers
import tensorflow as tf
import numpy as np
sess = tf.InteractiveSession()


# In[2]:

shuffle = np.load('data.npy')
from scipy.io import loadmat
data = loadmat("letters_data.mat")
train_x = data['train_x']

# y needs to be one hot encoded
old_y = data['train_y'] - 1
from scipy.io import loadmat
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

# In[3]:


def batch_generator(n):
    while True:
        shuffle = np.random.choice(np.arange(100000), replace=False, size=100000)
        for i in range(1000):
            yield tx[i * n: (i+1) * n], ty[i * n: (i+1) * n]


# In[4]:


x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, 26], name='y_')


# In[5]:


input_layer= tf.reshape(x, [-1, 28, 28, 1], name='input')


# In[6]:


# Conv Pool 1
conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      name='conv1')
pool1 = tf.layers.max_pooling2d(
    inputs=conv1,
    pool_size=[2, 2],
    strides=2,
    name='pool1')


# In[7]:


# Conv Pool 2
conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      name='conv2')
pool2 = tf.layers.max_pooling2d(
    inputs=conv2,
    pool_size=[2, 2],
    strides=2,
    name='pool2')


# In[8]:


# Dense layer 1
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64], name='pool2_flat')
dense1 = tf.layers.dense(
    inputs=pool2_flat,
    units=1024,
    activation=tf.nn.relu,
    name='dense1')
dropout1 = tf.layers.dropout(dense1, keep_prob, name='dropout1')


# In[9]:


# Dense layer 2
dense2 = tf.layers.dense(
    inputs=dropout1,
    units=512,
    activation=tf.nn.relu,
    name='dense2')
dropout2 = tf.layers.dropout(dense2, keep_prob, name='dropout2')


# In[10]:


# Logits
logits = tf.layers.dense(inputs=dropout2, units=26, name='logits')


# In[11]:


cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[12]:


# new instance
sess.run(tf.global_variables_initializer())


# In[13]:


batch_gen = batch_generator(100)
for i in range(200000):        
    batch = next(batch_gen)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: vx, y_: vy, keep_prob: 1.0}))


# In[14]:


# save
variables = [
    x,
    y_,
    input_layer,
    conv1,
    pool1,
    conv2,
    pool2,
    keep_prob,
    pool2_flat,
    dense1,
    dropout1,
    dense2,
    dropout2,
    logits
]
for var in variables:
    tf.add_to_collection('vars', var)
saver = tf.train.Saver()
saver.save(sess, 'neural-net-model')

