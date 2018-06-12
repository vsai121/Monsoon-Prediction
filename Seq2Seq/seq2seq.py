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

batch_size = 256

ratio = [1,1,1]

ratio = np.reshape(ratio , [-1,1])

train_loss = []
validation_loss = []

def generate_batches(batch_size , X_train , Y_train):

    num_batches = int(len(X_train)) // batch_size

    if(num_batches*batch_size< len(X_train)):
        num_batches+=0

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
    init_learning_rate = 0.001
    lambda_l2_reg = 0.0001
    max_epoch = 301
    learning_rate_decay = 0.99

    ## Network Parameters
    # length of input signals
    input_seq_len = l.NUM_STEPS

    # length of output signals
    output_seq_len = l.LEAD_TIME

    # size of LSTM Cell
    hidden_dim = 80

    dropout= [0]
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
    learning_rate = tf.placeholder(dtype=tf.float32, shape=None)
    dropout = tf.placeholder(dtype=tf.float32 , shape=(config.num_stacked_layers))

    return enc_inp , target_seq , dec_inp , learning_rate , dropout

def weight_variable(shape):
    return (tf.Variable(tf.truncated_normal(shape=shape , stddev=0.1)))

def bias_variable(shape):
    return tf.Variable(tf.constant(0., shape=shape))

def encoder_network(dropout):
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


    seq_lengths = tf.constant(config.output_seq_len,shape=[batch_size])

    helper = tf.contrib.seq2seq.TrainingHelper(
    decoder_inputs, seq_lengths, time_major=True)

    # Decoder
    decoder = tf.contrib.seq2seq.BasicDecoder(
    cell, helper,state)

    print("Decoder done")
    outputs, state,_ = tf.contrib.seq2seq.dynamic_decode(decoder)

    print("Outputs",outputs)
    
    """
    for i, inp in enumerate(decoder_inputs):

        if loop_function is not None and prev is not None:
           inp = loop_function(prev, Why , by )

        if i > 0:
            variable_scope.get_variable_scope().reuse_variables()

        output,state = cell(inp, state=state)
        outputs.append(output)


        if loop_function is not None:
            prev = output
    """
    return outputs[0], state

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
        i = dec_outputs[:,j,:]

        temp = (tf.add(tf.matmul(i , Why),by))
        reshaped_outputs.append(temp)

        last_outputs = temp

    return reshaped_outputs , last_outputs


def compute_loss(reshaped_outputs , last_outputs,target_seq , learning_rate):

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

        # L2 regularization for weights and biases
        reg_loss = 0
        for tf_var in tf.trainable_variables():
            reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

        loss = output_loss  + config.lambda_l2_reg* reg_loss

    with tf.variable_scope('Optimizer' , reuse=tf.AUTO_REUSE):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        minimize = optimizer.minimize(loss)


    return loss , optimizer , minimize





def build_train_graph(feed_previous = False):

    print("Building graph")
    global_step = step()

    Why = weight_variable([config.hidden_dim , config.output_dim])
    by = bias_variable([config.output_dim])


    print("Weights initialised")

    enc_inp , target_seq , dec_inp , learning_rate , dropout = create_placeholders()
    print("Placeholders created")

    cell = encoder_network(dropout)
    print("Encoder created")

    dec_outputs, dec_memory = _basic_rnn_seq2seq(enc_inp, dec_inp, cell, Why , by , feed_previous=feed_previous)
    print("decoder computed")

    reshaped_outputs,last_outputs = reshape(dec_outputs , Why , by)
    print("Outputs computed")

    loss , optimizer , minimize = compute_loss(reshaped_outputs , last_outputs,target_seq , learning_rate)
    print("Loss computed")

    return dict(
        enc_inp = enc_inp,
        target_seq = target_seq,
        reshaped_outputs = reshaped_outputs,
        loss = loss,
        learning_rate=learning_rate,
        dropout=dropout,
        optimizer = optimizer,
        minimize = minimize,
        )


def train():


    total_epochs = config.max_epoch
    train_losses = []

    rnn_model = build_train_graph(feed_previous=False)

    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        sess.run(init)

        checkpoint = tf.train.get_checkpoint_state("saved_networks")

        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Loaded :", checkpoint.model_checkpoint_path)
            print("\n\n")
        else:
            print("Unable to find network weights")


        print("Training losses: ")

        learning_rates = [
        config.init_learning_rate * (
            math.pow(config.learning_rate_decay , (i))
        ) for i in range(config.max_epoch)]

        for epoch_step in range(total_epochs):
            total_loss = 0
            avg_loss = 0
            j=0
            batch_inputs , batch_outputs = generate_batches(batch_size,X_train , y_train)

            for batch_input,batch_output in zip(batch_inputs , batch_outputs):

                current_lr = learning_rates[i]
                feed_dict = {rnn_model['enc_inp'][t]: batch_input[:,t] for t in range(config.input_seq_len)}
                feed_dict.update({rnn_model['target_seq'][t]: batch_output[:,t] for t in range(config.output_seq_len)})
                feed_dict.update({rnn_model['dropout']:config.dropout})
                feed_dict.update({rnn_model['learning_rate']:current_lr})


                outputs = sess.run(rnn_model['reshaped_outputs'] , feed_dict)

                loss_t= sess.run(rnn_model['loss'] , feed_dict)
                _ = sess.run(rnn_model['minimize'] , feed_dict)
                total_loss += loss_t
                j+=1

                #print("Outputs\n" , outputs),
                #print("Actual\n" ,  batch_output)
                #print("loss" , loss_t)

            avg_loss = total_loss/j
            print("Epoch " + str(epoch_step))
            print("Average loss "),
            print(avg_loss)

            if(epoch_step%5==0):
                saver.save(sess, 'saved_networks/' , global_step = epoch_step)

                print("Checkpoint saved")

                train_loss.append(avg_loss)

                batch_inputs , batch_outputs = generate_batches(batch_size,X_validation , y_validation)
                loss_v = 0
                j=0
                for batch_input,batch_output in zip(batch_inputs , batch_outputs):

                    feed_dict = {rnn_model['enc_inp'][t]: batch_input[:,t] for t in range(config.input_seq_len)}
                    feed_dict.update({rnn_model['target_seq'][t]: batch_output[:,t] for t in range(config.output_seq_len)})

                    dropout = [1 for i in range(config.num_stacked_layers)]
                    feed_dict.update({rnn_model['dropout']:dropout})
                    j+=1
                    loss_v += sess.run(rnn_model['loss'], feed_dict)


                print("Validation loss" , loss_v/j)

if __name__=='__main__':
    train()
