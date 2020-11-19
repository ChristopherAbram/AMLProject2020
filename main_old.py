import numpy as np
import sys, os

import tensorflow.compat.v1 as tf
import tensorflow as tf2
from tensorflow.keras import datasets
from datetime import datetime

tf.disable_v2_behavior()


class Autoencoder:

    def __init__(self, input_size, n_classes, file_writer):
        self.input_size = input_size
        self.n_classes = n_classes
        self.file_writer = file_writer

    def compile(self):
        self.x = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32)
        self.y = tf.placeholder(shape=[None, self.n_classes], dtype=tf.float32)
        self.omega = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32) # TODO: make generalized to be a list
        # s = tf.placeholder(shape=[None, input_size], dtype=tf.float32)
        
        # Hidden layers:
        W1 = tf.Variable(tf.random_normal(shape=[self.input_size, 200]))
        b1 = tf.Variable(tf.random_normal(shape=[200]))
        y1 = tf.nn.sigmoid(tf.matmul(self.x, W1) + b1) # first hidden layer (200 nodes)

        W2 = tf.Variable(tf.random_normal(shape=[200, 100]))
        b2 = tf.Variable(tf.random_normal(shape=[100]))
        y2 = tf.nn.sigmoid(tf.matmul(y1, W2) + b2) # second hidden layer / bottle neck (compressed) (100 nodes)

        W3 = tf.Variable(tf.random_normal(shape=[100, 200]))
        b3 = tf.Variable(tf.random_normal(shape=[200]))
        y3 = tf.nn.sigmoid(tf.matmul(y2, W3) + b3) # third hidden layer (200 nodes)
        
        # Output layer:
        W4 = tf.Variable(tf.random_normal(shape=[200, self.input_size]))
        b4 = tf.Variable(tf.random_normal(shape=[self.input_size]))
        self.x_r = tf.nn.sigmoid(tf.matmul(y3, W4) + b4) # output layer / reconstruction
        
        # Define reconstruction error (Equation 4)
        self.loss = tf.reduce_sum(tf.math.square(tf.norm(self.omega - self.x_r))) # PCA, TODO: add s variable
        
        self.learning_rate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate = self.learning_rate).minimize(self.loss) # TODO: stochastic gradient descent, eg (tf.keras.optimizers.SGD)

    def log_loss(self, value, step):
        tf.summary.scalar('loss',value)

    def fit(self, X, y, epochs=10000):
        init = tf.global_variables_initializer()
        with tf.Session() as session, self.file_writer:
            session.run(init)

            for epoch in range(epochs):
                
                _, loss_value = session.run([self.optimizer, self.loss], feed_dict={
                    self.x: X, 
                    self.y: y, 
                    self.omega: X, 
                    self.learning_rate: 0.001
                })

                summary = tf.summary.scalar('loss', loss_value)
                session.run(summary)

                # self.log_loss(loss_value, epoch)

                if epoch % 10 == 0:
                    # self.file_writer.flush()
                    print("Epoch: ", epoch, "loss_value=", loss_value)

            print("Done")

    def evaluate(self):
        pass


def main(argc, argv):
    n_epochs = 10000
    n_classes = 10
    n_samples = 1000 # reduce dataset
    max_val = 255

    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

    # Clear out any prior log data
    # os.system("rm -rf logs")

    logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    #file_writer = tf2.summary.create_file_writer(logdir)
    file_writer = tf.summary.FileWriter(logdir, flush_secs=1)
    #with file_writer.as_default():
    with file_writer:
        images = np.reshape(X_train[0:25], (-1, 28, 28, 1))
        tf.summary.image("Training data", images, max_outputs=25)
        
    # file_writer.flush()

    # reduce and normalise data
    X_train = np.array([X.ravel() for X in X_train[:n_samples]]) / max_val
    X_test = np.array([X.ravel() for X in X_test[:n_samples]]) / max_val

    input_size = X_train.shape[1]

    # Categorical representation of the labels:
    y_train, y_test = tf2.keras.utils.to_categorical(y_train, num_classes=n_classes), \
        tf2.keras.utils.to_categorical(y_test, num_classes=n_classes)

    model = Autoencoder(input_size, n_classes, file_writer)
    model.compile()
    model.fit(X_train, y_train, epochs=1000)

    # x = tf.placeholder(shape=[None, input_size], dtype=tf.float32)
    # y = tf.placeholder(shape=[None, n_classes], dtype=tf.float32)
    # omega = tf.placeholder(shape=[None, input_size], dtype=tf.float32) # TODO: make generalized to be a list
    # # s = tf.placeholder(shape=[None, input_size], dtype=tf.float32)
    
    # # Hidden layers:
    # W1 = tf.Variable(tf.random_normal(shape=[input_size, 200]))
    # b1 = tf.Variable(tf.random_normal(shape=[200]))
    # y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1) # first hidden layer (200 nodes)

    # W2 = tf.Variable(tf.random_normal(shape=[200, 100]))
    # b2 = tf.Variable(tf.random_normal(shape=[100]))
    # y2 = tf.nn.sigmoid(tf.matmul(y1, W2) + b2) # second hidden layer / bottle neck (compressed) (100 nodes)

    # W3 = tf.Variable(tf.random_normal(shape=[100, 200]))
    # b3 = tf.Variable(tf.random_normal(shape=[200]))
    # y3 = tf.nn.sigmoid(tf.matmul(y2, W3) + b3) # third hidden layer (200 nodes)
    
    # # Output layer:
    # W4 = tf.Variable(tf.random_normal(shape=[200, input_size]))
    # b4 = tf.Variable(tf.random_normal(shape=[input_size]))
    # x_r = tf.nn.sigmoid(tf.matmul(y3, W4) + b4) # output layer / reconstruction
    
    # # Define reconstruction error (Equation 4)
    # loss = tf.reduce_sum(tf.math.square(tf.norm(omega - x_r))) # PCA, TODO: add s variable
    
    # learning_rate = tf.placeholder(tf.float32)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss) # TODO: stochastic gradient descent, eg (tf.keras.optimizers.SGD)

    # # Variables for prediction and accuracy
    # prediction = tf.argmax(x_r, 1)
    # accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y, 1)), tf.float32))

    # Run training:
    # init = tf.global_variables_initializer()
    # with tf.Session() as session, file_writer.as_default():
    #     session.run(init)

    #     for epoch in range(n_epochs):
            
    #         _, loss_value = session.run([optimizer, loss], feed_dict={
    #             x: X_train, 
    #             y: y_train, 
    #             omega: X_train, 
    #             learning_rate: 0.001
    #         })

    #         tf2.summary.scalar('loss', data=loss_value, step=epoch)

    #         if epoch % 10 == 0:
    #             print("Epoch: ", epoch, "loss_value=", loss_value)


    #     print("Done")

        # accuracy_value = session.run(accuracy, feed_dict={x: X_train, y: y_train})
        # print("Accuracy:", accuracy_value)



    # # Defined the model parameters
    # W1 = tf.get_variable("W1", [2, 5], initializer=tf.random_normal_initializer)
    # b1 = tf.get_variable("b1", [5], initializer=tf.random_normal_initializer)
    # W2 = tf.get_variable("W2", [5, 2], initializer=tf.random_normal_initializer)
    # b2 = tf.get_variable("b2", [2], initializer=tf.random_normal_initializer)
        
    # # Construct model
    # a1 = tf.matmul(x, W1) + b1
    # z1 = tf.nn.tanh(a1)
    # a2 = tf.matmul(z1, W2) + b2
    # y = tf.nn.softmax(a2)

    # # Variables for prediction and accuracy
    # prediction = tf.argmax(y, 1)
    # accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(t, 1)), tf.float32))

    # # Define the loss function
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=t, logits=a2))

    # # Define the optimizer operation
    # learning_rate = tf.placeholder(tf.float32)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)

    # # Make an operation that initializes the variables
    # init = tf.global_variables_initializer()
    

    return 0


if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))