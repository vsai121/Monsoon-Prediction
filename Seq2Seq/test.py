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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


X_train , y_train ,  X_test , y_test , ratio = l.process()


ratio = [1,1,1]
ratio = np.reshape(ratio , [-1,1])

batch_size = len(y_test)

def write_list_to_file(list, filename):
    """Write the list to csv file."""

    with open(filename, "w") as outfile:
        for entries in list:
            outfile.write(str(entries))
            outfile.write("\n")


class RNNConfig():


    ## Network Parameters
    # length of input signals
    input_seq_len = l.NUM_STEPS

    # length of output signals
    output_seq_len = l.LEAD_TIME

    # size of LSTM Cell
    hidden_dim = 30

    # num of input signals
    input_dim = l.INPUTS

    # num of output signals
    output_dim = 3

    # num of stacked lstm layers
    num_stacked_layers = 1

    alpha = 1 #Loss parameter



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

def encoder_network():
    cells = []
    for i in range(config.num_stacked_layers):

        lstm_cell = tf.contrib.rnn.LSTMCell(config.hidden_dim)
        cells.append(lstm_cell)

    cell = tf.contrib.rnn.MultiRNNCell(cells)

    return cell

def decoder_network(attention_mechanism):

    lstm_cell = tf.contrib.rnn.LSTMCell(config.hidden_dim)

    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
        lstm_cell, attention_mechanism=attention_mechanism,
        attention_layer_size=config.hidden_dim)


    return decoder_cell


def _rnn_decoder(decoder_inputs,initial_state, attention_mechanism , Why , by , loop_function=None,scope=None):

    outputs = []
    prev = None

    cell = decoder_network(attention_mechanism)

    state = cell.zero_state(batch_size=batch_size,dtype=tf.float32).clone(cell_state=initial_state[0])

    print("Outputs",outputs)

    for i, inp in enumerate(decoder_inputs):

        if loop_function is not None and prev is not None:
           inp = loop_function(prev, Why , by )

        if i > 0:
            variable_scope.get_variable_scope().reuse_variables()

        output,state = cell(inp, state=state)
        outputs.append(output)


        if loop_function is not None:
            prev = output

    return outputs, state
def _basic_rnn_seq2seq(encoder_inputs,decoder_inputs,cell,Why , by , feed_previous, dtype=dtypes.float32,scope=None):

    enc_cell = copy.deepcopy(cell)
    outputs, enc_state = rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)

    enc_outputs = tf.stack(outputs)

    attention_states = tf.transpose(enc_outputs, [1, 0, 2])

    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                            num_units=config.hidden_dim, memory =attention_states)

    if feed_previous:
        return _rnn_decoder(decoder_inputs, enc_state, attention_mechanism , Why , by ,_loop_function)
    else:
        return _rnn_decoder(decoder_inputs, enc_state , attention_mechanism , Why , by)


def reshape(dec_outputs , Why , by):

    reshaped_outputs = []

    for j in range(config.output_seq_len):

        print("j",j)
        i = dec_outputs[j]

        temp = (tf.add(tf.matmul(i , Why),by))
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


            if(Y==target_seq[-1]):
                weight_per_label = tf.transpose(tf.matmul(Y,class_weight))
                last_step_cost = tf.reduce_mean(tf.multiply(weight_per_label,tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=y)))


        output_loss = config.alpha * all_steps_cost + (1-config.alpha) * last_step_cost

        return output_loss

def _loop_function(prev, Why , by):

    temp= tf.nn.softmax(tf.matmul(prev, Why) + by)
    one_hot = tf.one_hot(tf.argmax(temp, dimension = 1), depth = 3)
    return one_hot



def build_graph(feed_previous = True):

    print("Building graph")
    tf.reset_default_graph()

    global_step = step()


    Why = weight_variable([config.hidden_dim , config.output_dim])
    by = bias_variable([config.output_dim])
    print("Weights initialised")

    enc_inp , target_seq , dec_inp = create_placeholders()
    print("Placeholders created")

    cell = encoder_network()
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
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        sess.run(init)

        checkpoint = tf.train.get_checkpoint_state("saved_networks")

        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Loaded :", checkpoint.model_checkpoint_path)
        else:
            print("Unable to find network weights")


        global y_validation
        feed_dict = {rnn_model['enc_inp'][t]: X_test[:, t, :] for t in range(config.input_seq_len)} # batch prediction
        feed_dict.update({rnn_model['target_seq'][t]: y_test[:,t] for t in range(config.output_seq_len)})
        final_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)


        final_preds = [np.expand_dims(pred, 1) for pred in final_preds]
        final_preds = np.concatenate(final_preds, axis = 1)

        loss = sess.run(rnn_model['loss'] , feed_dict)

        preds=[]
        act=[]

        updated_pred = []
        updated_act = []


        for pred in final_preds:
            preds.append(pred[-1])



        for a in y_test:
            act.append(a[-1])


        for i in range(len(preds)):
            if((i+1)%122 > 30 and (i+1)%122 < 93 ):
                updated_pred.append(np.argmax(preds[i]))

        write_list_to_file(updated_pred, "Predictions.csv")

        for i in range(len(preds)):
            if((i+1)%122 > 30 and (i+1)%122 < 93 ):
                updated_act.append(np.argmax(act[i]))
        write_list_to_file(updated_act, "Actual.csv")

if __name__ == '__main__':
    test()        
