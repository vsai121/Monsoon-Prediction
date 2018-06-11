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
    output_dim= 3

    num_steps=l.NUM_STEPS
    output_size = l.LEAD_TIME

    lstm_size=[100]
    num_layers=len(lstm_size)


config = RNNConfig()

import tensorflow as tf
tf.reset_default_graph()
lstm_graph = tf.Graph()
def create_placeholders():

    inputs = tf.placeholder(dtype= tf.float32, shape = [None, config.num_steps, config.input_size])
    targets = tf.placeholder(dtype = tf.float32, shape = [None, config.output_size , config.output_dim ])


    return inputs , targets

def weight_variable(shape):
    return (tf.Variable(tf.truncated_normal(shape=shape , stddev=0.1)))

def bias_variable(shape):
    return tf.Variable(tf.constant(0., shape=shape))


def create_network():
    cells = []
    for i in range(config.num_layers):
        cell = tf.contrib.rnn.LSTMCell(config.lstm_size[i])  # Or LSTMCell(num_units)
        cells.append(cell)

    cell = tf.contrib.rnn.MultiRNNCell(cells)
    return cell

def init_params(inputs):
    cell = create_network()
    val,_ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    val = tf.transpose(val, [1, 0, 2])
    outputs = []
    #print(val.shape)
    for i in range(config.output_size):

        last = tf.gather(val, int(val.get_shape()[0]) - 1, name="last_lstm_output")
        outputs.append(last)
    #weight and bias between hidden and output layer
    Why = weight_variable([config.lstm_size[-1] , config.output_dim])
    by = bias_variable([config.output_dim])

    return outputs , Why , by

def compute_output(inputs):

    outputs , Why , by = init_params(inputs)

    prediction = []
    for i in range(config.output_size):
        prediction.append(tf.add(tf.matmul(outputs[i] , Why),by))

    return prediction


def compute_loss(prediction , targets):

    net = [v for v in tf.trainable_variables()]
    weight_reg = (tf.add_n([0.001 * tf.nn.l2_loss(var) for var in net]))

    all_steps_cost = 0
    last_step_cost=0

    for t in range(len(prediction)):

        y = prediction[t]
        Y = targets[:,t,:]

        print(y)
        print(Y)
        all_steps_cost += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=y))

        if(Y==targets[-1]):
            last_step_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=y))

    output_loss = 0.5 * all_steps_cost + (0.5) * last_step_cost

    return output_loss

def test(inputs , targets,sess):

    prediction = compute_output(inputs)
    loss = compute_loss(prediction,targets)

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
        targets:y_validation,
    }

    preds,loss = sess.run([prediction,loss] , validation_data_feed)


    #print("Preds" , preds)

    pred=[]
    act=[]

    """
    Calculating class from softmax predictions  Example - [0.1 , 0.8 , 0.1] = Class 1
    """
    for p,a in zip(preds[0] , y_validation):
        #print("Preds" , p),
        #print("Actual" , a)
        pred.append(np.argmax(p))
        act.append(np.argmax(a))

    """
    Calculating accuracy

    """
    acc=0
    count=0

    for i in range(len(pred)):
        if(act[i]!=1):
            count+=1
            if(pred[i]==act[i]):
                acc+=1

    print("Active/dry spell accuracy")
    print(float(acc)/count)*100


    acc=0
    count=0
    for i in range(len(pred)):
        if(1):
            count+=1
            if(pred[i]==act[i]):
                acc+=1

    print("Total accuracy")
    print(float(acc)/count)*100

    print("Loss",loss)

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
