
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

X_train , y_train ,X_validation , y_validation ,  X_test , y_test = l.process()


ratio = [1,1,1]
ratio = np.reshape(ratio , [-1,1])
class RNNConfig():


    ## Network Parameters
    # length of input signals
    input_seq_len = l.NUM_STEPS

    # length of output signals
    output_seq_len = l.LEAD_TIME

    # size of LSTM Cell
    hidden_dim = 80

    # num of input signals
    input_dim = l.INPUTS

    # num of output signals
    output_dim = 3

    # num of stacked lstm layers
    num_stacked_layers = 1



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
        cells.append(tf.contrib.rnn.GRUCell(config.hidden_dim))
    cell = tf.contrib.rnn.MultiRNNCell(cells)

    return cell

def _rnn_decoder(decoder_inputs,initial_state,cell, Why , by , loop_function=None,scope=None):

    state = initial_state
    outputs = []
    prev = None

    for i, inp in enumerate(decoder_inputs):

        if loop_function is not None and prev is not None:
           inp = loop_function(prev, Why , by)

        if i > 0:
            variable_scope.get_variable_scope().reuse_variables()

        output, state = cell(inp, state)
        outputs.append(output)

        if loop_function is not None:
            prev = output

    return outputs, state

def _basic_rnn_seq2seq(encoder_inputs,decoder_inputs,cell,Why , by , feed_previous, dtype=dtypes.float32,scope=None):

    enc_cell = copy.deepcopy(cell)
    _, enc_state = rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)

    if feed_previous:
        return _rnn_decoder(decoder_inputs, enc_state, cell, Why , by ,  _loop_function)
    else:
        return _rnn_decoder(decoder_inputs, enc_state, cell , Why , by)

def reshape(dec_outputs , Why , by):

    reshaped_outputs = []

    for i in dec_outputs:

        temp = tf.matmul(i , Why) + by
        reshaped_outputs.append(temp)

    return reshaped_outputs


def _loop_function(prev, Why , by):

    temp= (tf.nn.softmax(tf.matmul(prev, Why) + by))
    #temp = tf.one_hot(tf.argmax(temp, dimension = 1), depth = 3)

    return temp

def compute_loss(reshaped_outputs , target_seq):

    class_weight = tf.constant(ratio)
    class_weight = tf.cast(class_weight , tf.float32)
    # L2 loss
    output_loss = 0
    for _y, _Y in zip(reshaped_outputs, target_seq):
        weight_per_label = tf.transpose(tf.matmul(_Y, (class_weight)) )
        output_loss += tf.reduce_mean(tf.multiply(weight_per_label,tf.nn.softmax_cross_entropy_with_logits(logits=_y, labels=_Y)))
        #output_loss+= tf.reduce_mean(tf.square(_Y - _y))



    return output_loss


def build_graph(feed_previous = True):

    print("Building graph")
    tf.reset_default_graph()

    global_step = step()


    Why = weight_variable([config.hidden_dim , config.output_dim])
    by = bias_variable([config.output_dim])
    print("Weights initialised")

    enc_inp , target_seq , dec_inp = create_placeholders()
    print("Placeholders created")

    cell = create_network()
    print("Network created")

    dec_outputs, dec_memory = _basic_rnn_seq2seq(enc_inp, dec_inp, cell, Why , by , feed_previous=feed_previous)
    print("decoder computed")

    reshaped_outputs = reshape(dec_outputs, Why , by)
    print("Outputs computed")

    loss = compute_loss(reshaped_outputs , target_seq)

    return dict(
        enc_inp = enc_inp,
        target_seq = target_seq,
        reshaped_outputs = reshaped_outputs,
        loss=loss,
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
        feed_dict.update({rnn_model['target_seq'][t]: y_validation[:,t] for t in range(config.output_seq_len)})
        final_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)


        final_preds = [np.expand_dims(pred, 1) for pred in final_preds]
        final_preds = np.concatenate(final_preds, axis = 1)

        loss = sess.run(rnn_model['loss'] , feed_dict)

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


        acc=0
        count=0

        for i in range(len(preds)):
            if(act[i]!=1):
                count+=1
                if(preds[i]==act[i]):
                    acc+=1

        print("Active/dry spell accuracy")
        print(float(acc)/count)*100


        acc=0
        count=0
        for i in range(1,len(preds)):
            if(1):
                count+=1
                if(preds[i]==act[i]):
                    acc+=1

        print("Total accuracy")
        print(float(acc)/count)*100

        print("validation_loss" , loss)

        line1 = ax1.plot(preds,'bo-',label='list 1')
        line2 = ax1.plot(act,'go-',label='list 2')

        # Display the figure
        plt.show()



test()
