import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes

    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    X = tf.placeholder(tf.float32, shape=[None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, shape=[None, n_y])

    return X, Y

def initialize_parameters(filters):
    """
    Initializes weight parameters to build cnn with tensorflow using filters
    
    Returns:
    parameters -- a dictionary of tensors containing weights
    """

    # filters = [[4,4,3,8],[2,2,8,16]]
    tf.set_random_seed(1)
    parameters = {}

    for l in range(len(filters)):
        parameters["W"+str(l+1)] = tf.get_variable("W"+str(l+1),
                                                 shape=filters[l],
                                                 initializer=tf.contrib.layers.xavier_initializer(seed = 0))

    return parameters


def forward_propagation(X, parameters, c_strides, p_strides, p_ksizes, n_y, padding="SAME"):
    """
    Implements the forward propagation for the model:
    (L-1)[CONV2D -> RELU -> MAXPOOL] -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X -- input dataset placeholder, of shape (number of examples, n_H0, n_W0, n_C0)
    parameters -- python dictionary containing your parameters

    Returns:
    ZL -- the output of the last LINEAR unit, of shape (n_y, number of examples)
    """

    P = X
    for l in range(len(parameters)):

        W = parameters["W"+str(l+1)]
        c_stride = c_strides["stride"+str(l+1)]
        p_stride = p_strides["stride"+str(l+1)]
        p_ksize = p_ksizes["ksize"+str(l+1)]

        Z = tf.nn.conv2d(P,
                         W,
                         strides = [1, c_stride, c_stride, 1],
                         padding = padding)
        A = tf.nn.relu(Z)
        P = tf.nn.max_pool(A,
                           ksize = [1, p_ksize, p_ksize, 1],
                           strides = [1, p_stride, p_stride, 1],
                           padding = padding)

    P = tf.contrib.layers.flatten(P)

    # maintain linear activation function by setting it explicitly to None
    # default: ReLU, which we dont want. Softmax will be applied in cost function
    Z = tf.contrib.layers.fully_connected(P, num_outputs = n_y, activation_fn=None)

    return Z


def compute_cost(ZL, Y):
    """
    Computes the cost

    Arguments:
    ZL -- output of forward propagation (output of the last LINEAR unit), of shape (n_y, number of examples)
    Y -- "true" labels vector placeholder, same shape as ZL

    Returns:
    cost - Tensor of the cost function
    """

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=ZL))
    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
          num_epochs = 100, minibatch_size = 64, print_cost = True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X_train -- training set, of shape (None, n_H0, n_W0, n_C0)
    Y_train -- test set, of shape (None, n_y)
    X_test -- training set, of shape (None, n_H0, n_W0, n_C0)
    Y_test -- test set, of shape (None, n_y)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 5 epochs

    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)  # to keep consistent results
    seed = 3  # to keep consistent results
    (m, n_H0, n_W0, n_C0) = X_train.shape
    costs = []
    n_y = Y_train.shape[1]

    # hyper-parameters
    filters = [[4,4,3,8],[2,2,8,16]]
    c_strides = {"stride1": 1,
                 "stride2": 1}
    p_strides = {"stride1": 8,
                 "stride2": 4}
    p_ksizes = {"ksize1": 8,
                "ksize2": 4}

    # 1. Create placeholders
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    # 2. Initialize parameters
    parameters = initialize_parameters(filters)

    # 3. Forward propagation
    ZL = forward_propagation(X,
                             parameters,
                             c_strides,
                             p_strides,
                             p_ksizes,
                             n_y,
                             padding="SAME")

    # 4. Compute cost
    cost = compute_cost(ZL, Y)

    # 5. Create optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # initialize variables
    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        sess.run(init)
        for epoch in range(num_epochs):

            epoch_cost = 0
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1 # because it shouldnt generate same minibatches in every run
            minibatches = random_mini_batches(X_train, Y_train, mini_batch_size=minibatch_size, seed=seed)

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch

                _, minibatch_cost = sess.run([optimizer, cost],feed_dict={X: minibatch_X,
                                                      Y: minibatch_Y})

                epoch_cost += (minibatch_cost / num_minibatches)

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(epoch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.savefig('images/cost.png')

        parameters = sess.run(parameters)
        correct_predictions = tf.equal(tf.argmax(ZL,1), tf.argmax(Y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions,"float"))

        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})

        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)


        # NOTE: optimizer <- cost <- ZL <- forward_prop <- initialize_params <- create_placeholders <- X,Y
        # In the above chain, X is fed only with X_train (in mini-batches) for training
        # backprop runs during training

        # NOTE: accuracy <- correct_predictions <- ZL <- forward_prop <- initialize_params <- create_placeholders <- X,Y
        # In the above chain, X is fed with both X_train and X_test
        # backprop doesn't run during training (only forward prop)

        return train_accuracy, test_accuracy, parameters