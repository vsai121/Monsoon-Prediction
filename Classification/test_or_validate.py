import tensorflow as tf
import numpy as np

import random
import loader as l

import matplotlib.pyplot as plt
from scipy import spatial

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

import math


X_train , y_train ,X_validation , y_validation ,  X_test , y_test  = l.process()

BATCH_SIZE = 256 #BATCH GRADIENT DESCENT FOR TRAINING

def generate_batches(batch_size , X , Y):

    num_batches = int(len(X)) // batch_size

    if batch_size * num_batches < len(X):
        num_batches += 1


    batch_indices = range(num_batches)

    #print("batch_indices" , batch_indices)
    batches_X = []
    batches_Y = []

    for j in batch_indices:
        batch_X =  X[j * batch_size: (j + 1) * batch_size]
        batch_y =  Y[j * batch_size: (j + 1) * batch_size]

        batches_X.append(batch_X)
        batches_Y.append(batch_y)

    return batches_X , batches_Y

class RNNConfig():

    input_size=l.INPUTS
    output_size = 3
    num_steps=l.NUM_STEPS
    lstm_size=[15]
    num_layers=len(lstm_size)
    keep_prob=1

config = RNNConfig()

import tensorflow as tf
tf.reset_default_graph()
lstm_graph = tf.Graph()

def create_placeholders():
    inputs = tf.placeholder(dtype= tf.float32, shape = [None, config.num_steps, config.input_size])
    targets = tf.placeholder(dtype = tf.float32, shape = [None, config.output_size ])
    return inputs,targets

def weight_variable(shape):
    return (tf.Variable(tf.random_normal(shape=shape , stddev=0.1)))

def bias_variable(shape):
    return tf.Variable(tf.constant(0., shape=shape))


def create_network():
    cells = []
    for i in range(config.num_layers):
        cell = tf.contrib.rnn.LSTMCell(config.lstm_size[i])  # Or LSTMCell(num_units)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1)
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



def test(inputs , targets,sess):

    prediction = compute_output(inputs)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=targets, name="xent_raw"))
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init)

    checkpoint = tf.train.get_checkpoint_state("saved_networks")

    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Loaded :", checkpoint.model_checkpoint_path)
    else:
        print("Unable to find network weights")


    batches_X , batches_y = generate_batches(BATCH_SIZE , X_validation, y_validation)

    preds = []
    act = []
    temp=[]
    k=0
    for batch_X, batch_y in zip(batches_X, batches_y):
        #print(batch_X)
        #print(batch_y)
        #print("\n\n\n")
        validation_data_feed = {
            inputs: batch_X,
            targets : batch_y
        }
        #print(batch_X , batch_y)

        pred = sess.run(prediction , validation_data_feed)
        validation_loss = sess.run(loss , validation_data_feed)

        print(validation_loss)


        for p in pred:
            preds.append(p)

        for a in batch_y:
            act.append(a)




    #regr = linear_model.LinearRegression()

# Train the model using the training sets
    #temp = np.reshape(preds , [-1,1])
    #regr.fit(temp, act)

    #preds = [p*float(regr.coef_) + float(regr.intercept_) for p in preds]

    cost = 0
    count = 0
    acc = 0
    for i in range(len(preds)):

        print("Prediction" , preds[i]),
        print("Actual" , act[i])

        preds[i] = np.argmax(preds[i])
        act[i] = np.argmax(act[i])

        print("Prediction" , preds[i]),
        print("Actual" , act[i])


        if(act[i]!=1 and act[i]!=2):
            cost+=1
            if(preds[i]==act[i]):
                count+=1
        if(preds[i]==act[i]):
            acc+=1

    print(float(count)/cost)
    print(float(acc)/len(preds))
    fig = plt.figure()



# Make room for legend at bottom

    fig.subplots_adjust(bottom=0.2)


    # The axes for your lists 1-3
    ax1 = fig.add_subplot(111)
    # Plot lines 1-3
    line1 = ax1.plot(preds,'bo-',label='list 1')
    line2 = ax1.plot(act,'go-',label='list 2')

    # Display the figure
    plt.show()

def process():

    print("Testing  hahaxD")
    sess = tf.InteractiveSession()
    inp,targets  = create_placeholders()
    test(inp , targets, sess)

if __name__ == '__main__':
    process()
