import tensorflow as tf
import numpy as np

import random
import loader as l

import matplotlib.pyplot as plt
from scipy import spatial



X_train , y_train ,X_validation , y_validation ,  X_test , y_test = l.process()

BATCH_SIZE = 256 #BATCH GRADIENT DESCENT FOR TRAINING

def generate_batches(batch_size , X_train , Y_train):

    num_batches = int(len(X_train)) // batch_size

    if batch_size * num_batches < len(X_train):
        num_batches += 1


    batch_indices = range(num_batches)

    #print("batch_indices" , batch_indices)
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
    output_size =l.LEAD_TIME
    num_steps=l.NUM_STEPS
    lstm_size=8
    num_layers=1
    keep_prob=1
    batch_size = 256


config = RNNConfig()

import tensorflow as tf
tf.reset_default_graph()
lstm_graph = tf.Graph()

def create_placeholders():
    inputs = tf.placeholder(dtype= tf.float32, shape = [None, config.num_steps, config.input_size])

    return inputs

def weight_variable(shape):
    return (tf.Variable(tf.truncated_normal(shape=shape)))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def create_one_cell():

    return tf.contrib.rnn.LSTMCell(config.lstm_size, state_is_tuple=True)

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
    Why = weight_variable([config.lstm_size , config.output_size])
    by = bias_variable([config.output_size])

    return last , Why , by

def compute_output(inputs):

    last , Why , by = init_params(inputs)
    prediction = tf.matmul(last, Why) + by

    return prediction



def test(inputs ,sess):

    prediction = compute_output(inputs)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init)

    checkpoint = tf.train.get_checkpoint_state("saved_networks")

    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Loaded :", checkpoint.model_checkpoint_path)
    else:
        print("Unable to find network weights")


    batches_X , batches_y = generate_batches(BATCH_SIZE , X_train, y_train)

    preds = []
    act = []
    for batch_X, batch_y in zip(batches_X, batches_y):
        #print(batch_X)
        #print(batch_y)
        #print("\n\n\n")
        validation_data_feed = {
            inputs: batch_X,
        }
        #print(batch_X , batch_y)

        pred = sess.run(prediction , validation_data_feed)

        #print(train_loss)
        for p in pred:
            preds.append(p[28])

        for a in batch_y:
            act.append(a[28])

    fig = plt.figure()
    for i in range(len (preds)):
        cost = [abs(a_i - b_i) for a_i, b_i in zip(preds, act)]
    print(sum(cost)/len(cost))

    print("Preds" , preds)
    print("Actual" , act)
# Make room for legend at bottom
    fig.subplots_adjust(bottom=0.2)

    # The axes for your lists 1-3
    ax1 = fig.add_subplot(111)

    # Plot lines 1-3
    line1 = ax1.plot(preds[0:50],'bo-',label='list 1')
    line2 = ax1.plot(act[0:50],'go-',label='list 2')




    # Display the figure
    plt.show()

if __name__== "__main__":
    print("Testing  xD")
    sess = tf.InteractiveSession()
    inp  = create_placeholders()
    test(inp ,  sess)
