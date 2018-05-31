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

batch_size = 512

ratio = [1.,1.,1.]

ratio = np.reshape(ratio , [-1,1])

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
    init_learning_rate = 0.0001
    lambda_l2_reg = 0.0003
    max_epoch = 400
    learning_rate_decay = 0.99

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
    learning_rate = tf.placeholder(dtype=tf.float32, shape=None)

    return enc_inp , target_seq , dec_inp , learning_rate


def create_network():
    cells = []
    for i in range(config.num_stacked_layers):
        with tf.variable_scope('RNN_{}'.format(i)):
            cells.append(tf.contrib.rnn.LSTMCell(config.hidden_dim , activation=tf.nn.relu))

    cells.append(tf.contrib.rnn.LSTMCell(config.output_dim , activation = tf.nn.softmax))
    cell = tf.contrib.rnn.MultiRNNCell(cells)

    return cell

def _rnn_decoder(decoder_inputs,initial_state,cell, loop_function=None,scope=None):

    state = initial_state
    outputs = []
    prev = None

    if loop_function is not None and prev is not None:
       inp = loop_function(prev)

    for i, inp in enumerate(decoder_inputs):

        output, state = cell(inp, state)
        outputs.append(output)

        if loop_function is not None:
            prev = output

    return outputs, state

def _basic_rnn_seq2seq(encoder_inputs,decoder_inputs,cell, feed_previous, dtype=dtypes.float32,scope=None):

    enc_cell = copy.deepcopy(cell)
    _, enc_state = rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)

    if feed_previous:
        return _rnn_decoder(decoder_inputs, enc_state, cell , _loop_function)
    else:
        return _rnn_decoder(decoder_inputs, enc_state, cell)

def _loop_function(prev):
    return prev;


def compute_loss(reshaped_outputs , target_seq , learning_rate):

     # Training loss and optimizer
    with tf.variable_scope('Loss'):
        # L2 loss
        output_loss = 0
        class_weight = tf.constant(ratio)
        class_weight = tf.cast(class_weight , tf.float32)

        for y, Y in zip(reshaped_outputs, target_seq):
            output_loss += tf.reduce_mean(tf.square(y-Y))

        # L2 regularization for weights and biases
        reg_loss = 0
        for tf_var in tf.trainable_variables():
            reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

        loss = output_loss + config.lambda_l2_reg * reg_loss

    with tf.variable_scope('Optimizer' , reuse=tf.AUTO_REUSE):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -2.5, 2.5), var) for grad, var in gvs]
        minimize = optimizer.apply_gradients(capped_gvs)


    return loss , optimizer , minimize





def build_graph(feed_previous = False):

    print("Building graph")
    tf.reset_default_graph()

    global_step = step()

    enc_inp , target_seq , dec_inp , learning_rate = create_placeholders()
    print("Placeholders created")

    cell = create_network()
    print("Network created")

    dec_outputs, dec_memory = _basic_rnn_seq2seq(enc_inp, dec_inp, cell, feed_previous=feed_previous)
    print("decoder computed")

    reshaped_outputs = [i for i in dec_outputs]
    print("Outputs computed")

    loss , optimizer , minimize = compute_loss(reshaped_outputs , target_seq , learning_rate)
    print("Loss computed")


    train_losses = []


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


        print("Training losses: ")

        learning_rates = [
        config.init_learning_rate * (
            math.pow(config.learning_rate_decay , (i/10))
        ) for i in range(config.max_epoch)]

        for epoch_step in range(config.max_epoch):
            total_loss = 0
            avg_loss = 0
            j = 0

            batch_inputs , batch_outputs = generate_batches(batch_size,X_train , y_train)
            for batch_input,batch_output in zip(batch_inputs , batch_outputs):
                current_lr = learning_rates[epoch_step]
                feed_dict = {enc_inp[t]: batch_input[:,t] for t in range(config.input_seq_len)}
                feed_dict.update({target_seq[t]: batch_output[:,t] for t in range(config.output_seq_len)})

                feed_dict.update({learning_rate:current_lr})
                pred = sess.run(reshaped_outputs , feed_dict);
                loss_t,_ = sess.run([loss , minimize], feed_dict)
                total_loss += loss_t
                j+=1
                #print("pred" , pred[-1])

            avg_loss = total_loss/j

            print("Epoch " + str(epoch_step))
            print("Average loss "),
            print(avg_loss)

            if epoch_step%10==0:
                saver.save(sess, 'saved_networks/' , global_step = epoch_step)

                print("Checkpoint saved")


print("Hello world")
build_graph()
