import tensorflow as tf
import numpy as np

import random
import loader as l

import math
import matplotlib.pyplot as plt

from tensorflow.contrib import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes
import copy

X_train , y_train ,X_validation , y_validation ,  X_test , y_test,_,_,_ = l.process()


def generate_batches(batch_size , X_train , Y_train):

    num_batches = int(len(X_train)) // batch_size

    if batch_size * num_batches < len(X_train):
        num_batches += 1


    batch_indices = range(num_batches)


    random.shuffle(batch_indices)
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


    ## Parameters
    learning_rate = 0.001
    lambda_l2_reg = 0.001

    ## Network Parameters
    # length of input signals
    input_seq_len = l.NUM_STEPS

    # length of output signals
    output_seq_len = l.LEAD_TIME

    # size of LSTM Cell
    hidden_dim = 64

    # num of input signals
    input_dim = l.INPUTS

    # num of output signals
    output_dim = 1

    # num of stacked lstm layers
    num_stacked_layers = 2

    # gradient clipping - to avoid gradient exploding
    GRADIENT_CLIPPING = 2.5

config = RNNConfig()


def step():
    global_step = tf.Variable(initial_value=0,name="global_step",trainable=False,
    collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    return global_step

def create_placeholders():
    enc_inp = [tf.placeholder(tf.float32, shape=(None, config.input_dim), name="inp_{}".format(t))for t in range(config.input_seq_len)]
    target_seq = [tf.placeholder(tf.float32, shape=(None, config.output_dim), name="y".format(t))for t in range(config.output_seq_len)]
    dec_inp = [ tf.zeros_like(target_seq[0], dtype=tf.float32, name="GO") ] + target_seq[:-1]

    return enc_inp , target_seq , dec_inp

def weight_variable(shape):
    return (tf.Variable(tf.truncated_normal(shape=shape , stddev=0.1)))

def bias_variable(shape):
    return tf.Variable(tf.constant(0., shape=shape))


def create_network():
    cells = []
    for i in range(config.num_stacked_layers):
        with tf.variable_scope('RNN_{}'.format(i)):
            cells.append(tf.contrib.rnn.LSTMCell(config.hidden_dim))
    cell = tf.contrib.rnn.MultiRNNCell(cells)

    return cell

def _rnn_decoder(decoder_inputs,initial_state,cell,loop_function=None,scope=None):

    state = initial_state
    outputs = []
    prev = None

    for i, inp in enumerate(decoder_inputs):

        if loop_function is not None and prev is not None:
           inp = loop_function(prev, i)

        if i > 0:
            variable_scope.get_variable_scope().reuse_variables()

        output, state = cell(inp, state)
        outputs.append(output)

        if loop_function is not None:
            prev = output

    return outputs, state

def _basic_rnn_seq2seq(encoder_inputs,decoder_inputs,cell,feed_previous,dtype=dtypes.float32,scope=None):

    enc_cell = copy.deepcopy(cell)
    _, enc_state = rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)

    if feed_previous:
        return _rnn_decoder(decoder_inputs, enc_state, cell, _loop_function)
    else:
        return _rnn_decoder(decoder_inputs, enc_state, cell)

def _loop_function(prev, _):
    return tf.matmul(prev, weights['out']) + biases['out']


def compute_loss(reshaped_outputs , target_seq):

    output_loss = 0
    for _y, _Y in zip(reshaped_outputs, target_seq):
        output_loss += tf.reduce_mean(tf.pow(_y - _Y, 2))

    reg_loss = 0
    for tf_var in tf.trainable_variables():
        reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

    loss = output_loss + config.lambda_l2_reg * reg_loss
    optimizer = tf.contrib.layers.optimize_loss(loss=loss,learning_rate=config.learning_rate,global_step=step(),optimizer='Adam',
                                                clip_gradients=config.GRADIENT_CLIPPING)

    return loss , optimizer


def build_graph(feed_previous = False):

    tf.reset_default_graph()

    global_step = step()

    Why = weight_variable([config.hidden_dim , config.output_dim])
    by = bias_variable([config.output_dim])

    enc_inp , target_seq , dec_inp = create_placeholders()

    cell = create_network()

    dec_outputs, dec_memory = _basic_rnn_seq2seq(enc_inp, dec_inp, cell, feed_previous = feed_previous)

    reshaped_outputs = [tf.matmul(i, Why) + by for i in dec_outputs]

    loss , optimizer = compute_loss(reshaped_outputs , target_seq)

    total_epochs = 100
    batch_size = 72
    KEEP_RATE = 0.5
    train_losses = []



    rnn_model = build_graph(feed_previous=False)

    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        sess.run(init)

        print("Training losses: ")
        for i in range(total_epochs):

            batch_inputs , batch_outputs = generate_batches(batch_size,X_train , y_train)
            for batch_input,batch_output in zip(batch_inputs , batch_outputs):

                feed_dict = {enc_inp[t]: batch_input[:,t] for t in range(input_seq_len)}
                feed_dict.update({target_seq[t]: batch_output[:,t] for t in range(output_seq_len)})

                _, loss_t = sess.run([loss , optimizer], feed_dict)
                print(loss_t)

                saver.save(sess, 'saved_networks/' , global_step = i)

                print("Checkpoint saved")

if __name__== "__main__":
    build_graph()
