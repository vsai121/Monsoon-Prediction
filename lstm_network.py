import tensorflow as tf
import numpy as np

import random
import loader as l



BATCH_SIZE = 80 #BATCH GRADIENT DESCENT FOR TRAINING

X_train , y_train ,X_validation , y_validation ,  X_test , y_test = l.process()

def generate_batches(batch_size , X_train , Y_train , validation_phase):

    num_batches = int(len(X_train)) // batch_size

    if batch_size * num_batches < len(X_train):
        num_batches += 1


    batch_indices = range(num_batches)

    if not validation_phase:
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

    input_size=1
    num_steps=60
    lstm_size=256
    num_layers=2
    keep_prob=0.8
    batch_size = 50
    init_learning_rate = 0.03
    learning_rate_decay = 0.99
    init_epoch = 5
    max_epoch = 1000

config = RNNConfig()

import tensorflow as tf
tf.reset_default_graph()
lstm_graph = tf.Graph()

def create_placeholders():
    inputs = tf.placeholder(dtype= tf.float32, shape = [None, config.num_steps, config.input_size])
    targets = tf.placeholder(dtype = tf.float32, shape = [None, config.input_size])
    learning_rate = tf.placeholder(dtype=tf.float32, shape=None)

    return inputs , targets , learning_rate

def weight_variable(shape):
    return (tf.Variable(tf.truncated_normal(shape=shape)))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def create_one_cell():

    return tf.contrib.rnn.LSTMCell(config.lstm_size, state_is_tuple=True)
    if config.keep_prob < 1.0:
        return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)

def multiple_layers():

    if config.num_layers >1:
        cell = tf.contrib.rnn.MultiRNNCell([create_one_cell() for _ in range(config.num_layers)],state_is_tuple=True)

    else:
        cell = create_one_cell()

    return cell

def init_params(inputs):
    cell = multiple_layers()
    val,_ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

    #print(val.shape)
    val = tf.transpose(val, [1, 0, 2])
    last = tf.gather(val, int(val.get_shape()[0]) - 1, name="last_lstm_output")

    #weight and bias between hidden and output layer
    Why = weight_variable([config.lstm_size , config.input_size])
    by = bias_variable([config.input_size])

    return last , Why , by

def compute_output(inputs):

    last , Why , by = init_params(inputs)
    prediction = tf.matmul(last, Why) + by

    return prediction

def compute_loss(prediction , targets , learning_rate):

    loss = tf.reduce_mean(tf.square(prediction - targets))

    optimizer = tf.train.AdagradOptimizer(learning_rate)
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



    tf.global_variables_initializer().run()
    learning_rates = [
    config.init_learning_rate * (
        config.learning_rate_decay ** max(float(i + 1 - config.init_epoch), 0.0)
    ) for i in range(config.max_epoch)]


    predictions = []
    for epoch_step in range(config.max_epoch):
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

            train_loss, _ = sess.run([loss, minimize], train_data_feed)
            total_loss+=train_loss
            #print(train_loss)

            j+=1

        print("Epoch" +str(epoch_step)+ "completed")
        average_loss = total_loss/j
        print("Average loss for this epoch is  " + str(average_loss))
        print("\n")

        #print("\n\n\n\n\n")

        if(epoch_step % 40==0 and epoch_step>0):
            print("Saving state")
            saver = tf.train.Saver()
            saver.save(sess, 'saved_networks/' , global_step = epoch_step)


if __name__== "__main__":
    print("Training haha xD")
    sess = tf.InteractiveSession()
    inp , output , learning_rate = create_placeholders()
    train(inp , output , learning_rate , sess)
    print(X_test.shape)
    print(y_test.shape)
    #test(X_test , y_test)
