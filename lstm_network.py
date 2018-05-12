import tensorflow as tf
import numpy as np

import random
import loader as l

BATCH_SIZE = 20 #BATCH GRADIENT DESCENT FOR TRAINING

X_train , y_train ,X_validation , y_validation ,  X_test , y_test = l.process()

def generate_batches(batch_size , X_train , Y_train):

    num_batches = int(len(X_train)) // batch_size

    if batch_size * num_batches < len(X_train):
        num_batches += 1

    batch_indices = range(num_batches)
    random.shuffle(batch_indices)

    batches_X = []
    batches_Y = []

    for j in batch_indices:
        batch_X =  X_train[j * batch_size: (j + 1) * batch_size]
        batch_y =  Y_train[j * batch_size: (j + 1) * batch_size]

        batches_X.append(batch_X)
        batches_Y.append(batch_y)

    return batches_X , batches_Y

class RNNConfig():

    input_size=1
    num_steps=30
    lstm_size=512
    num_layers=2
    keep_prob=0.8
    batch_size = 20
    init_learning_rate = 0.001
    learning_rate_decay = 0.99
    init_epoch = 5
    max_epoch = 50

config = RNNConfig()

import tensorflow as tf
tf.reset_default_graph()
lstm_graph = tf.Graph()

def create_placeholders():
    inputs = tf.placeholder(dtype= tf.float32, shape = [None, config.num_steps, config.input_size])
    targets = tf.placeholder(dtype = tf.float32, shape = [None, config.input_size])
    learning_rate = tf.placeholder(dtype=tf.float32, shape=None)

    return inputs , targets , learning_rate

def weight_variable(shape):
    return (tf.Variable(tf.truncated_normal(shape=shape)))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def create_one_cell():

    return tf.contrib.rnn.LSTMCell(config.lstm_size, state_is_tuple=True)
    if config.keep_prob < 1.0:
        return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)

def multiple_layers():

    if config.num_layers >1:
        cell = tf.contrib.rnn.MultiRNNCell([create_one_cell() for _ in range(config.num_layers)],state_is_tuple=True)

    else:
        cell = create_one_cell()

    return cell

def init_params(inputs):
    cell = multiple_layers()
    val,_ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

    #print(val.shape)
    val = tf.transpose(val, [1, 0, 2])
    last = tf.gather(val, int(val.get_shape()[0]) - 1, name="last_lstm_output")

    #weight and bias between hidden and output layer
    Why = weight_variable([config.lstm_size , config.input_size])
    by = bias_variable([config.input_size])

    return last , Why , by

def compute_output(inputs):

    last , Why , by = init_params(inputs)
    prediction = tf.matmul(last, Why) + by

    return prediction

def compute_loss(prediction , targets , learning_rate):

    loss = tf.reduce_mean(tf.square(prediction - targets))

    optimizer = tf.train.AdamOptimizer(learning_rate)
    minimize = optimizer.minimize(loss)

    return loss , optimizer , minimize

def train():


    inputs , targets , learning_rate = create_placeholders()
    prediction = compute_output(inputs)
    loss , optimizer , minimize = compute_loss(prediction , targets , learning_rate)

    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        learning_rates = [
        config.init_learning_rate * (
            config.learning_rate_decay ** max(float(i + 1 - config.init_epoch), 0.0)
        ) for i in range(config.max_epoch)]

        i = 0
        for epoch_step in range(config.max_epoch):
            current_lr = learning_rates[epoch_step]
            total_loss = 0
            j = 0
            batches_X , batches_y = generate_batches(BATCH_SIZE , X_train , y_train)


            for batch_X, batch_y in zip(batches_X, batches_y):
                train_data_feed = {
                    inputs: batch_X,
                    targets: batch_y,
                    learning_rate: current_lr
                }
                #print(batch_X , batch_y)
                train_loss, _ = sess.run([loss, minimize], train_data_feed)
                total_loss+=train_loss
                print(train_loss )
                j+=1

            print("Epoch" + str(i) +   "completed\n")
            average_loss = total_loss/j
            print("Average loss for this epoch is " + str(average_loss))
            print("\n\n\n\n\n")
            i+=1
        saver = tf.train.Saver()
        saver.save(sess, 'saved_networks/' , global_step = epoch_step)



if __name__== "__main__":
    train()
    print(X_test.shape)
    print(y_test.shape)
    test(X_test , y_test)
