import numpy as np
import sys, os

import tensorflow as tf
from tensorflow.keras import datasets, optimizers
from datetime import datetime

# tf.python.framework.ops.disable_eager_execution()

if(tf.executing_eagerly()):
    print('Eager execution is enabled (running operations immediately)\n')
else:
    print('Eager execution is off\n')

print(('\nYour devices that are available:\n{0}').format(
    [device.name for device in tf.config.experimental.list_physical_devices()]
))

class DeepGAE:

    def __init__(self, input_size, n_classes, file_writer):
        self.input_size = input_size
        self.n_classes = n_classes
        self.file_writer = file_writer
        self.parameter_list = None

    @tf.function
    def predict(self, X):
        y1 = self.layer1(X)
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)
        return self.layer4(y3)

    def compile(self, optimizer):
        self.optimizer = optimizer
        X_init = tf.Variable(np.random.normal(size=(1, self.input_size)), dtype=tf.float32)

        # # Hidden layers:
        self.W1 = tf.Variable(tf.random.normal(shape=[self.input_size, 200]), name='w1')
        self.b1 = tf.Variable(tf.random.normal(shape=[200]), name='b1')
        # self.y1 = tf.nn.sigmoid(tf.matmul(X_init, self.W1) + self.b1)
        self.layer1 = lambda X: tf.nn.sigmoid(X @ self.W1 + self.b1) # first hidden layer (200 nodes)

        self.W2 = tf.Variable(tf.random.normal(shape=[200, 100]), name='w2')
        self.b2 = tf.Variable(tf.random.normal(shape=[100]), name='b2')
        # self.y2 = tf.nn.sigmoid(tf.matmul(self.y1, self.W2) + self.b2)
        self.layer2 = lambda X: tf.nn.sigmoid(X @ self.W2 + self.b2) # second hidden layer / bottle neck (compressed) (100 nodes)

        self.W3 = tf.Variable(tf.random.normal(shape=[100, 200]), name='w3')
        self.b3 = tf.Variable(tf.random.normal(shape=[200]), name='b3')
        # self.y3 = tf.nn.sigmoid(tf.matmul(self.y2, self.W3) + self.b3)
        self.layer3 = lambda X: tf.nn.sigmoid(X @ self.W3 + self.b3) # third hidden layer (200 nodes)

        # Output layer:
        self.W4 = tf.Variable(tf.random.normal(shape=[200, self.input_size]), name='w4')
        self.b4 = tf.Variable(tf.random.normal(shape=[self.input_size]), name='b4')
        # self.y4 = tf.nn.sigmoid(tf.matmul(self.y3, self.W4) + self.b4)
        self.layer4 = lambda X: tf.nn.sigmoid(X @ self.W4 + self.b4) # output layer / reconstruction

        # Init logits:
        # self.predict(X_init)

        # if self.parameter_list is None:
            # Init references, only once:
        self.parameter_list = [
            self.W1, self.b1,
            self.W2, self.b2,
            self.W3, self.b3,
            self.W4, self.b4,
        ]

        # Define reconstruction error (Equation 4)
        # self.loss = tf.reduce_sum(tf.math.square(tf.norm(self.omega - self.x_r))) # PCA, TODO: add s variable
        
        # self.learning_rate = tf.placeholder(tf.float32)
        # self.optimizer = tf.train.GradientDescentOptimizer(
        #     learning_rate = self.learning_rate).minimize(self.loss) # TODO: stochastic gradient descent, eg (tf.keras.optimizers.SGD)

    def __loss(self, X, X_pred):
        return tf.reduce_mean(tf.square(tf.norm(X - X_pred)))

    def __gradients(self, input, target):
        with tf.GradientTape() as tape:
            tape.watch(self.parameter_list)
            loss_value = self.__loss(input, target)
            grads = tape.gradient(tf.constant(1.5), self.parameter_list)
            return loss_value, grads

    def fit(self, X, y, epochs=10000):
        with self.file_writer.as_default():
            for epoch in range(epochs):
                loss_value = 0
                for X_batch, y_batch in X:

                    X_pred = self.predict(X_batch)
                    # loss = self.__loss(X_batch, X_pred)
                    # loss_value = loss.numpy()

                    loss = lambda: tf.reduce_sum(tf.math.square(tf.norm(X_batch - X_pred)))

                    # loss_value, grads = self.__gradients(X_batch, X_pred)
                    # self.optimizer.apply_gradients(zip(grads, self.parameter_list))
                    self.optimizer.minimize(loss, self.parameter_list)

                if epoch % 10 == 0:
                    tf.summary.scalar('loss', loss_value, step=epoch)
                    print("Epoch: ", epoch, "loss_value=", loss_value)




        # init = tf.global_variables_initializer()
        # with tf.Session() as session, self.file_writer:
        #     session.run(init)

        #     for epoch in range(epochs):
                
        #         _, loss_value = session.run([self.optimizer, self.loss], feed_dict={
        #             self.x: X, 
        #             self.y: y, 
        #             self.omega: X, 
        #             self.learning_rate: 0.001
        #         })

        #         summary = tf.summary.scalar('loss', loss_value)
        #         session.run(summary)

        #         # self.log_loss(loss_value, epoch)

        #         if epoch % 10 == 0:
        #             # self.file_writer.flush()
        #             print("Epoch: ", epoch, "loss_value=", loss_value)

        #     print("Done")

    def evaluate(self):
        pass

def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 2550.0
    x = tf.reshape(x, [28*28])
    y = tf.cast(y, tf.int32)
    return x, y

def dataset(batch_size, limit_train_samples=0):
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
    if limit_train_samples > 0:
        X_train = X_train[:limit_train_samples]
        y_train = y_train[:limit_train_samples]

    def _dataset(X, y):
        y = tf.one_hot(y, depth=10)
        X = tf.data.Dataset.from_tensor_slices((X, y)) \
            .map(preprocess) \
            .shuffle(10000) \
            .batch(batch_size)
        return X, y

    X_train, y_train = _dataset(X_train, y_train)
    X_test, y_test = _dataset(X_test, y_test)
    return (X_train, y_train), (X_test, y_test)


def main(argc, argv):
    n_epochs = 10000
    n_classes = 10
    batch_size = 8
    n_samples = 1000 # reduce dataset

    (X_train, y_train), (X_test, y_test) = dataset(batch_size, n_samples)

    logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    file_writer = tf.summary.create_file_writer(logdir)
    # with file_writer.as_default():
    #     images = np.reshape(X_train[0:25], (-1, 28, 28, 1))
    #     tf.summary.image("Training data", images, max_outputs=25)
        
    input_size = 28 * 28

    model = DeepGAE(input_size, n_classes, file_writer)
    # Note! We use keras optimizer.
    # TODO: try with momentum
    model.compile(optimizers.SGD(learning_rate=0.01))
    model.fit(X_train, y_train, epochs=n_epochs)


        


    # reduce and normalise data
    # X_train = np.array([X.ravel() for X in X_train[:n_samples]]) / max_val
    # X_test = np.array([X.ravel() for X in X_test[:n_samples]]) / max_val

    # input_size = X_train.shape[1]
    # model.compile()
    # model.fit(X_train, y_train, epochs=1000)

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

    return 0


if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))