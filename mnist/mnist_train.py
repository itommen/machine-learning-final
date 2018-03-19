import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import matplotlib.pyplot as plt

def gnerateLayers():
    n_input = 784
    n_hidden = 256
    n_second_hidden = 128
    n_output = 10

    y_true = tf.placeholder(tf.float32, [None, n_output])
    net_input = tf.placeholder(tf.float32, [None, n_input], name="input")    

    W = tf.Variable(tf.truncated_normal ([n_input, n_hidden]), name="w")
    b = tf.Variable(tf.truncated_normal ([n_hidden]), name="b")

    w2 = tf.Variable(tf.truncated_normal ([n_hidden, n_second_hidden]), name="w2")
    b2 = tf.Variable(tf.truncated_normal ([n_second_hidden]), name="b2")

    w3 = tf.Variable(tf.truncated_normal ([n_second_hidden, n_output]), name="w3")        
    b3 = tf.Variable(tf.truncated_normal ([n_output]), name="b3")
    
    prob = tf.placeholder_with_default(1.0, shape=())
    net_output = tf.nn.softmax(tf.nn.sigmoid(tf.matmul(tf.nn.dropout(tf.nn.sigmoid(tf.matmul(tf.nn.sigmoid(tf.matmul(net_input, W) + b), w2) + b2), prob), w3) + b3), name="output") # <-- THIS IS OUR MODEL!

    return net_input, net_output, prob, y_true

def printGraph(title, yLabel, xLabel, trainList, validationList):
    plt.title(title)
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.plot(trainList, color='b')
    plt.plot(validationList, color='r')
    plt.show()

def getOptimizer(cost):
    eta = 0.005
    return tf.train.AdamOptimizer(eta).minimize(cost)

def calcTrainData(func, net_input, y_true):
    train = sess.run(func, feed_dict={
                           net_input: mnist.train.images,
                           y_true: mnist.train.labels })

    validation = sess.run(func, feed_dict={
                        net_input: mnist.validation.images,
                        y_true: mnist.validation.labels })

    return train, validation

def trainModal(net_input, y_true, prob, cost, accuracy):
    l_accuracies_train = list()
    l_accuracies_validation = list()
    l_cost_train = list()
    l_cost_valid = list()

    batch_size = 200
    n_epochs = 30
    for epoch_i in range(n_epochs):
        for batch_i in range(0, mnist.train.num_examples, batch_size):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={
                net_input: batch_xs,
                y_true: batch_ys,
                prob: 0.55
            })        

        # Accuracy grpahes data          
        accuracyTrain, accuracyValidation = calcTrainData(accuracy, net_input, y_true)  
        print('Validation accuracy for train epoch {} is: {}'.format(epoch_i + 1, accuracyTrain))
        l_accuracies_train.append(accuracyTrain)

        print('Validation accuracy for validation epoch {} is: {}'.format(epoch_i + 1, accuracyValidation))
        l_accuracies_validation.append(accuracyValidation)        
        
        # Loss Graphes data
        lossTrain, lossValidation = calcTrainData(cost, net_input, y_true)
        print('Validation cost for train epoch {} is: {}'.format(epoch_i + 1, lossTrain))
        l_cost_train.append(lossTrain)

        print('Validation cost for valid epoch {} is: {}'.format(epoch_i + 1, lossValidation))
        l_cost_valid.append(lossValidation)
    
    return l_accuracies_train, l_accuracies_validation, l_cost_train, l_cost_valid

def saveModal(sess):
    modal_file_name = "digits_modal"
    saver = tf.train.Saver()    
    saver.save(sess, "./{}.ckpt".format(modal_file_name))

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

net_input, net_output, prob, y_true = gnerateLayers()

correct_prediction = tf.equal(tf.argmax(net_output, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=net_output, labels=y_true))

optimizer = getOptimizer(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())    

    l_accuracies_train, l_accuracies_validation, l_cost_train, l_cost_valid =  trainModal(net_input, y_true, prob, cost, accuracy)

    saveModal(sess)    

    printGraph('Train (blue) Validation (red)'
        , 'Accuracy'
        , 'Epochs'
        , l_accuracies_train
        , l_accuracies_validation)
    print("acc train: ", l_accuracies_train)
    print("acc validation: ", l_accuracies_validation)
    
    printGraph('Train (blue) Validation (red)'
        ,'Lost'
        , 'Epochs'
        , l_cost_train
        , l_cost_valid)

    print(l_cost_train, l_cost_valid)
    
    print("Accuracy for test set: {}". format(sess.run(accuracy,
                   feed_dict={
                       net_input: mnist.test.images,
                       y_true: mnist.test.labels
                   })))
