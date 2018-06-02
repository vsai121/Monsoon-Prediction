import tensorflow as tf
import numpy as np

import random
import loader2 as l

import matplotlib.pyplot as plt


import math

from tensorflow.contrib import rnn

"""

Same as training but takes X_validation or X_test as input and generates predictions

"""

X_train , y_train ,X_validation , y_validation ,  X_test , y_test  = l.process()

class RNNConfig():

    input_size=l.INPUTS
    output_size = 3
    num_steps=l.NUM_STEPS
    lstm_size=[40]
    num_layers=len(lstm_size)


config = RNNConfig()

import tensorflow as tf
tf.reset_default_graph()
lstm_graph = tf.Graph()

def create_placeholders():
    inputs = tf.placeholder(dtype= tf.float32, shape = [None, config.num_steps, config.input_size])
    targets = tf.placeholder(dtype = tf.float32, shape = [None , config.output_size ])
    return inputs,targets

def create_network():
    cells = []
    for i in range(config.num_layers):
        cell = tf.contrib.rnn.LSTMCell(config.lstm_size[i] , activation=tf.nn.relu)  # Or LSTMCell(num_units)
        cells.append(cell)

    cells.append( tf.contrib.rnn.LSTMCell(config.output_size , activation=tf.nn.softmax))
    cell = tf.contrib.rnn.MultiRNNCell(cells)
    return cell

def init_params(inputs):
    cell = create_network()
    val,_ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

    #print(val.shape)
    val = tf.transpose(val, [1, 0, 2])
    last = tf.gather(val, int(val.get_shape()[0]) - 1, name="last_lstm_output")

    return last

def compute_output(inputs):

    last = init_params(inputs)
    prediction = last

    return prediction



def test(inputs , targets,sess):

    prediction = compute_output(inputs)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init)


    """

    Checking for already saved weights.

    If found will resume with those weights

    Else start from beginning


    """


    checkpoint = tf.train.get_checkpoint_state("saved_networks")

    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Loaded :", checkpoint.model_checkpoint_path)
    else:
        print("Unable to find network weights")



    """
    Feeds X_validation to model and generates predictions

    """
    validation_data_feed = {
        inputs: X_validation,
    }

    preds = sess.run(prediction , validation_data_feed)

    #print("Preds" , preds)

    pred=[]
    act=[]

    """
    Calculating class from softmax predictions  Example - [0.1 , 0.8 , 0.1] = Class 1
    """
    for p,a in zip(preds , y_validation):
        print("Preds" , p),
        print("Actual" , a)
        pred.append(np.argmax(p))
        act.append(np.argmax(a))

    """
    Calculating accuracy

    """
    acc=0
    for i in range(len(pred)):
        if(pred[i]==act[i]):
            acc+=1

    print(float(acc)/len(pred)*100)


    """
    Plotting graph

    """

    fig = plt.figure()
    # Make room for legend at bottom

    fig.subplots_adjust(bottom=0.2)


    # The axes for your lists 1-3
    ax1 = fig.add_subplot(111)

    # Plot lines 1-3
    line1 = ax1.plot(pred,'bo-',label='list 1')
    line2 = ax1.plot(act,'go-',label='list 2')

    # Display the figure
    plt.show()



def process():

    print("Testin")
    sess = tf.InteractiveSession()
    inp,targets  = create_placeholders()
    test(inp , targets, sess)

if __name__ == '__main__':
    process()
