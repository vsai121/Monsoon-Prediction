
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

from sklearn.metrics import f1_score
X_train , y_train ,  X_test , y_test = l.process()


ratio = [1.4,1,1.4]
ratio = np.reshape(ratio , [-1,1])

batch_size = len(y_test)


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

    alpha = 0.5 #Loss parameter



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
    return tf.get_variable("proj_w_out",
    [config.hidden_dim, config.output_dim],dtype=tf.float32,initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))

def bias_variable(shape):
    return tf.get_variable("proj_b_out",
        [config.output_dim],dtype=tf.float32,initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))

def create_network():
    cells = []

    cell = tf.contrib.rnn.LSTMCell(config.hidden_dim)
    cells.append(cell)

    for j in range(config.num_stacked_layers-1):
        cell = tf.contrib.rnn.LSTMCell(config.hidden_dim)
        #cell = tf.contrib.rnn.ResidualWrapper(cell)
        cells.append(cell)

    cell = tf.contrib.rnn.MultiRNNCell(cells)

    return cell


def _rnn_decoder(decoder_inputs,initial_state,cell, Why , by , loop_function=None,scope=None):

    state = initial_state
    outputs = []
    prev = None

    for i, inp in enumerate(decoder_inputs):

        if loop_function is not None and prev is not None:
           inp = loop_function(prev, Why , by )

        if i > 0:
            variable_scope.get_variable_scope().reuse_variables()

        output, state = cell(inp, state)
        outputs.append(output)


        if loop_function is not None:
            prev = output

    return outputs, state

def _basic_rnn_seq2seq(encoder_inputs,decoder_inputs,cell,Why , by , feed_previous, dtype=dtypes.float32,scope=None):

    enc_cell = copy.deepcopy(cell)

    outputs, enc_state = rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)
    if feed_previous:
        return _rnn_decoder(decoder_inputs, enc_state, cell, Why , by ,_loop_function)
    else:
        return _rnn_decoder(decoder_inputs, enc_state, cell , Why , by)

def reshape(dec_outputs , Why , by):

    reshaped_outputs = []

    for j in range(config.output_seq_len):

        print("j",j)
        i = dec_outputs[j]
        temp = tf.matmul(i,Why)+by
        reshaped_outputs.append(temp)

        last_outputs = temp

    return reshaped_outputs , last_outputs

def compute_loss(reshaped_outputs , last_outputs,target_seq):

     # Training loss and optimizer
    with tf.variable_scope('Loss'):

        class_weight = tf.constant(ratio)
        class_weight = tf.cast(class_weight , tf.float32)


        # L2 loss
        output_loss = 0

        all_steps_cost = 0


        for y , Y in zip(reshaped_outputs , target_seq):
            weight_per_label = tf.transpose(tf.matmul(Y,class_weight))
            all_steps_cost += tf.reduce_mean(tf.multiply(weight_per_label,tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=y)))
            last_step_cost = tf.reduce_mean(tf.multiply(weight_per_label,tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=y)))


        output_loss = config.alpha * all_steps_cost + (1-config.alpha) * last_step_cost

        return output_loss



def _loop_function(prev, Why , by):

    print("prev",prev)
    temp = tf.add(tf.matmul(prev,Why),by)
    one_hot = tf.one_hot(tf.argmax(temp, dimension = 1), depth = 3)

    return one_hot


def build_graph(feed_previous = False):

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

    reshaped_outputs , last_outputs = reshape(dec_outputs, Why , by)
    print("Outputs computed")

    loss = compute_loss(reshaped_outputs ,last_outputs,target_seq)

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
        feed_dict = {rnn_model['enc_inp'][t]: X_test[:, t] for t in range(config.input_seq_len)} # batch prediction
        feed_dict.update({rnn_model['target_seq'][t]: y_test[:,t] for t in range(config.output_seq_len)})
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


        for a in y_test:
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

        print("F1 score" , f1_score(preds, act, average='weighted'))

        print("validation_loss" , loss)

        line1 = ax1.plot(preds,'bo-',label='list 1')
        line2 = ax1.plot(act,'go-',label='list 2')

        # Display the figure
        plt.show()



test()
