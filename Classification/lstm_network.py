import tensorflow as tf
import numpy as np

import random
import loader2 as l

import math


from tensorflow.contrib import rnn


#Loading the train , validation and test data
X_train , y_train ,X_validation , y_validation ,  X_test , y_test = l.process()


BATCH_SIZE =  72 # Batch size for batch gradient descent



def generate_batches(batch_size , X_train , Y_train , validation_phase):

    """
        Generating random mini-batches for batch gradient descent

    """
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

    """

    Network Hyper-parameters for the model

    """
    input_size=l.INPUTS # Number of variables
    output_size = 3  # 3 for 3 classes

    num_steps=l.NUM_STEPS #Days used to make prediction

    lstm_size=[40] #Size of LSTM cell
    dropout = [1]
    num_layers=len(lstm_size)  #Number of stacked layers in LSTM model

    """
    Learning Hyper-parameters

    """
    init_learning_rate = 0.0001  #Initial learning rate
    learning_rate_decay = 0.99  #Decay of learning rate
    lamda = 0.001
    max_epoch = 2000  #Total epochs


config = RNNConfig()

tf.reset_default_graph()
lstm_graph = tf.Graph()

def create_placeholders():

    """
    Creating placeholders for input , output and learning rate

    """
    inputs = tf.placeholder(dtype= tf.float32, shape = [None, config.num_steps, config.input_size])
    targets = tf.placeholder(dtype = tf.float32, shape = [None, config.output_size])
    learning_rate = tf.placeholder(dtype=tf.float32, shape=None)

    return inputs , targets , learning_rate

def create_network():

    """
    Create LSTM model
    """
    cells = []
    for i in range(config.num_layers):
        cell = tf.contrib.rnn.LSTMCell(config.lstm_size[i] , activation=tf.nn.relu)  # Or LSTMCell(num_units)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.dropout[i])
        cells.append(cell)

    """

    Append softmax layer to end for classification

    """

    cells.append( tf.contrib.rnn.LSTMCell(config.output_size , activation=tf.nn.softmax))

    cell = tf.contrib.rnn.MultiRNNCell(cells)
    return cell

def init_params(inputs):
    """
    Calling network with inputs to get output
    """
    cell = create_network()
    val,_ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

    #print(val.shape)
    val = tf.transpose(val, [1, 0, 2])
    last = tf.gather(val, int(val.get_shape()[0]) - 1, name="last_lstm_output")


    return last

def compute_output(inputs):

    """
    Computing output (no changes for classification)
    """
    last  = init_params(inputs)
    prediction = last

    return prediction

def compute_loss(prediction , targets , learning_rate):

    """
    Compute loss
    """

    """
    L2 regularization loss for avoiding overfitting
    """
    net = [v for v in tf.trainable_variables()]
    weight_reg = (tf.add_n([0.001 * tf.nn.l2_loss(var) for var in net]))

    """
    Softmax loss  = -1*target*log(prediction)
    """
    output_loss = 0
    output_loss +=  tf.reduce_mean(tf.multiply(targets,-tf.log(prediction)))

    """
    Adding the L2 regularization loss
    """

    loss = output_loss + config.lamda * weight_reg


    """
    Using adam optimizer
    """
    optimizer = tf.train.AdamOptimizer(learning_rate)
    minimize = optimizer.minimize(loss)

    return loss , optimizer , minimize

def train(inputs , targets , learning_rate , sess):

    """
        Training process
    """

    prediction = compute_output(inputs)
    loss , optimizer , minimize = compute_loss(prediction , targets , learning_rate)

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
    Computes decaying learning rate at each step
    learning rate(timestep t) = initital leaning rate * (decay_factor^t)

    """

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

        """
        batches _X and batches_Y are random shuffled batches of X_TRAIN AND Y_TRAIN
        """
        batches_X , batches_y = generate_batches(BATCH_SIZE , X_train , y_train , 0)


        for batch_X, batch_y in zip(batches_X, batches_y):

            """
            Computes loss for each batch

            """
            train_data_feed = {
                inputs: batch_X,
                targets: batch_y,
                learning_rate: current_lr
            }


            """
            Printing to check if prediction and loss are being computed correctly
            """

            #print(batch_X , batch_y)
            #print("batch_X" , batch_X)
            #print("batch_y" , batch_y)
            #pred = sess.run(prediction ,  train_data_feed)
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

        """
        Saves state of model every 10 epochs in saved_networks/ folder

        """
        if epoch_step%10==0:
            print("Saving state")
            saver = tf.train.Saver()
            saver.save(sess, 'saved_networks/' , global_step = epoch_step)
            #t.test(inputs , prediction ,  sess , saver)

if __name__== "__main__":

    """
    Calling tthe training process
    """

    sess = tf.InteractiveSession()
    inp , output , learning_rate = create_placeholders()
    train(inp , output , learning_rate , sess)
