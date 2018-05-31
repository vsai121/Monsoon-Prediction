
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


class RNNConfig():


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
    output_dim = 3

    # num of stacked lstm layers
    num_stacked_layers = 2



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


def create_network():
    cells = []
    for i in range(config.num_stacked_layers):
        with tf.variable_scope('RNN_{}'.format(i)):
            cells.append(tf.contrib.rnn.LSTMCell(config.hidden_dim , activation = tf.nn.relu))

    cells.append(tf.contrib.rnn.LSTMCell(config.output_dim , activation = tf.nn.softmax))
    cell = tf.contrib.rnn.MultiRNNCell(cells)

    return cell


def _rnn_decoder(decoder_inputs,initial_state,cell, loop_function=None,scope=None):

    state = initial_state
    outputs = []
    prev = None

    for i, inp in enumerate(decoder_inputs):

        if loop_function is not None and prev is not None:
           inp = loop_function(prev)

        if i > 0:
            variable_scope.get_variable_scope().reuse_variables()

        output, state = cell(inp, state)
        outputs.append(output)

        if loop_function is not None:
            prev = output

    return outputs, state

def _basic_rnn_seq2seq(encoder_inputs,decoder_inputs,cell,feed_previous, dtype=dtypes.float32,scope=None):

    enc_cell = copy.deepcopy(cell)
    _, enc_state = rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)

    if feed_previous:
        return _rnn_decoder(decoder_inputs, enc_state, cell,  _loop_function)
    else:
        return _rnn_decoder(decoder_inputs, enc_state, cell)

def _loop_function(prev):
    return prev


def build_graph(feed_previous = True):

    print("Building graph")
    tf.reset_default_graph()

    global_step = step()


    enc_inp , target_seq , dec_inp = create_placeholders()
    print("Placeholders created")

    cell = create_network()
    print("Network created")

    dec_outputs, dec_memory = _basic_rnn_seq2seq(enc_inp, dec_inp, cell, feed_previous=feed_previous)
    print("decoder computed")

    reshaped_outputs = [i for i in dec_outputs]
    print("Outputs computed")


    return dict(
        enc_inp = enc_inp,
        target_seq = target_seq,
        reshaped_outputs = reshaped_outputs,
        )

def test():

    rnn_model = build_graph(feed_previous=True)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        sess.run(init)

        checkpoint = tf.train.get_checkpoint_state("saved_networks")

        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Loaded :", checkpoint.model_checkpoint_path)
        else:
            print("Unable to find network weights")


        global y_validation
        feed_dict = {rnn_model['enc_inp'][t]: X_validation[:, t, :] for t in range(config.input_seq_len)} # batch prediction
        feed_dict.update({rnn_model['target_seq'][t]: np.zeros([y_validation.shape[0], config.output_dim], dtype=np.float32) for t in range(config.output_seq_len)})
        final_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)

        final_preds = [np.expand_dims(pred, 1) for pred in final_preds]
        final_preds = np.concatenate(final_preds, axis = 1)

        loss = tf.nn



        fig = plt.figure()
        fig.subplots_adjust(bottom=0.2)

        ax1 = fig.add_subplot(111)
        preds=[]
        act=[]
        for pred in final_preds:
            preds.append(pred[-1])


        for a in y_validation:
            act.append(a[-1])

        for a , b in zip(preds , act):
            print("Predicted   %d " , a),
            print("Actual  %d" , b)




        for i in range(len(preds)):
            preds[i] = np.argmax(preds[i])

        for i in range(len(preds)):
            act[i] = np.argmax(act[i])


        acc=0.
        count=0.
        for i in range(len(preds)):
            if(act[i]!=1):
                count+=1
                if(preds[i]==act[i]):
                    acc+=1

        print(float(acc)/count*100)

        line1 = ax1.plot(preds,'bo-',label='list 1')
        line2 = ax1.plot(act,'go-',label='list 2')

        # Display the figure
        plt.show()



test()
