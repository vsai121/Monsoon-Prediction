import tensorflow as tf
import numpy as np

import random
import loader2 as l

import math


from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

#BATCH GRADIENT DESCENT FOR TRAINING

X_train , y_train ,X_validation , y_validation ,  X_test , y_test = l.process()
BATCH_SIZE =  122

ratio = [1.25,1,1]

ratio = np.reshape(ratio , [-1,1])

def generate_batches(batch_size , X_train , Y_train , validation_phase):

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

    input_size=l.INPUTS
    output_size = 3
    num_steps=l.NUM_STEPS
    lstm_size=[10]
    num_layers=len(lstm_size)
    batch_size = 256
    init_learning_rate = 0.001
    learning_rate_decay = 0.99
    init_epoch = 5
    max_epoch = 2000

config = RNNConfig()

import tensorflow as tf
tf.reset_default_graph()
lstm_graph = tf.Graph()

def create_placeholders():
    inputs = tf.placeholder(dtype= tf.float32, shape = [None, config.num_steps, config.input_size])
    targets = tf.placeholder(dtype = tf.float32, shape = [None, config.output_size ])
    learning_rate = tf.placeholder(dtype=tf.float32, shape=None)

    return inputs , targets , learning_rate

def weight_variable(shape):
    return (tf.Variable(tf.truncated_normal(shape=shape , stddev=0.05)))

def bias_variable(shape):
    return tf.Variable(tf.constant(0., shape=shape))



def create_network():
    cells = []
    for i in range(config.num_layers):
        cell = tf.contrib.rnn.LSTMCell(config.lstm_size[i])  # Or LSTMCell(num_units)
        #cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.9)
        cells.append(cell)

    cell = tf.contrib.rnn.MultiRNNCell(cells)
    return cell

def init_params(inputs):
    cell = create_network()
    val,_ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

    #print(val.shape)
    val = tf.transpose(val, [1, 0, 2])
    last = tf.gather(val, int(val.get_shape()[0]) - 1, name="last_lstm_output")

    #weight and bias between hidden and output layer
    Why = weight_variable([config.lstm_size[-1] , config.output_size])
    by = bias_variable([config.output_size])

    return last , Why , by

def compute_output(inputs):

    last , Why , by = init_params(inputs)
    prediction = tf.matmul(last, Why) + by

    return prediction

def compute_loss(prediction , targets , learning_rate):

    net = [v for v in tf.trainable_variables()]
    weight_reg = (tf.add_n([0.001 * tf.nn.l2_loss(var) for var in net]))

    print("Weight_reg" , weight_reg)
    class_weight = tf.constant(ratio)
    class_weight = tf.cast(class_weight , tf.float32)

    xent = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=targets, name="xent_raw") + weight_reg
    loss = tf.reduce_mean(xent)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    minimize = optimizer.minimize(loss)

    return loss , optimizer , minimize

def train(inputs , targets , learning_rate , sess):

    prediction = compute_output(inputs)
    loss , optimizer , minimize = compute_loss(prediction , targets , learning_rate)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init)

    checkpoint = tf.train.get_checkpoint_state("saved_networks")

    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Loaded :", checkpoint.model_checkpoint_path)
    else:
        print("Unable to find network weights")

    learning_rates = [
    config.init_learning_rate * (
        math.pow(config.learning_rate_decay , (i))
    ) for i in range(config.max_epoch)]


    predictions = []
    for epoch_step in range(config.max_epoch):
        print("Epoch" , epoch_step)
        #print("\n\n\n")

        current_lr = learning_rates[epoch_step]
        total_loss = 0
        j = 0
        batches_X , batches_y = generate_batches(BATCH_SIZE , X_train , y_train , 0)


        for batch_X, batch_y in zip(batches_X, batches_y):

            train_data_feed = {
                inputs: batch_X,
                targets: batch_y,
                learning_rate: current_lr
            }
            #print(batch_X , batch_y)
            #print("batch_X" , batch_X)
            #print("batch_y" , batch_y)
            pred = sess.run(prediction ,  train_data_feed)
            #print("prediction" , pred)

            train_loss, _ = sess.run([loss, minimize], train_data_feed)
            #print("train_loss" , train_loss)

            #print("\n\n\n")
            total_loss+=train_loss
            #print(train_loss)

            j+=1

        print("Total loss" , total_loss/j)
        print("Epoch" +str(epoch_step)+ "completed")
        print("\n")

        #print("\n\n\n\n\n")

        if epoch_step%10==0:
            print("Saving state")
            saver = tf.train.Saver()
            saver.save(sess, 'saved_networks/' , global_step = epoch_step)
            #t.test(inputs , prediction ,  sess , saver)

if __name__== "__main__":
    print("multiple features haha chutiya 128 xD")
    sess = tf.InteractiveSession()
    inp , output , learning_rate = create_placeholders()
    train(inp , output , learning_rate , sess)
