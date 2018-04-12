import tensorflow as tf
import pandas as pd
import numpy as np
TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

def maybe_download():
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    return train_path, test_path

def load_data(y_name='Species'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    train_path, test_path = maybe_download()

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


###########################################################################

(train_x, train_y), (test_x, test_y) = load_data()
#import the trainning and testing data

train_y=list(train_y.values)
train_x=np.array(train_x).tolist()

test_y=list(test_y.values)
test_x=np.array(test_x).tolist()
#make model

NUM=3
N=10000
LR=0.001

def evaluate(logits,labels):
    pre=tf.argmax(logits,axis=1)
    wr=tf.reduce_mean(tf.cast(tf.equal(pre,labels),tf.float32))
    return pre, wr

x=tf.placeholder(tf.float32,shape=[None,4])
label=tf.placeholder(tf.int64,shape=[None])
label_o=tf.one_hot(label,NUM,1.0,0.0,-1)

dense1=tf.layers.Dense(units=16,activation=tf.nn.relu)
d1=dense1(x)
dense2=tf.layers.Dense(units=8)
d2=dense2(d1)
#the result is very bad if a use two actived layers
#if d2 does not have active function , the result seems to be better than one layer
logit_layer=tf.layers.Dense(units=3)
logit=logit_layer(d2)

loss=tf.losses.sparse_softmax_cross_entropy(labels=label,logits=logit)
op=tf.train.GradientDescentOptimizer(LR)
train=op.minimize(loss)

pre,wr= evaluate(logit,label)

#the test part
t_x=tf.placeholder(tf.float32,shape=[None,4])
t_label=tf.placeholder(tf.int64,shape=[None])
t_logit=logit_layer(dense2(dense1(t_x)))
pp,t_wr=evaluate(t_logit,t_label)

#run the model

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(N):
        _,my_loss,wrr=sess.run((train,loss,wr),feed_dict={x:train_x, label:train_y})
        if i%500==0:print(i,":",my_loss)
    test_wr=sess.run(t_wr,feed_dict={t_x:test_x,t_label:test_y})
    print(test_wr)


