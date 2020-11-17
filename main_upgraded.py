import numpy as np
import sys, os
import tensorflow as tf
from tensorflow.keras import datasets
from datetime import datetime

class ReconstructionError(tf.keras.losses.Loss):
    def __init__(self):
        super(ReconstructionError, self).__init__(name="reconstructionerror")

    def call(self, omega, x_r):
        return tf.reduce_sum(tf.math.square(tf.norm(omega - x_r))) # PCA, TODO: add s variable


class AutoencoderLayer(tf.keras.layers.Layer):
    def __init__(self, input_size, nodes):
        super(AutoencoderLayer, self).__init__()
        self.input_size = input_size
        self.nodes = nodes

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=[self.input_size, self.nodes],
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=[self.nodes], 
            initializer="random_normal", 
            trainable=True
        )

    def call(self, x):
        return tf.nn.sigmoid(tf.matmul(x, self.w) + self.b)

class Autoencoder(tf.keras.layers.Layer):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        self.firstHiddenLayer = AutoencoderLayer(input_size, 200)
        self.secondHiddenLayer = AutoencoderLayer(200, 100)
        self.thirdHiddenLayer = AutoencoderLayer(100, 200)
        self.outputLayer = AutoencoderLayer(200, input_size)

    def call(self, input):
        y1 = self.firstHiddenLayer(input)
        y2 = self.secondHiddenLayer(y1)
        y3 = self.thirdHiddenLayer(y2)
        y4 = self.outputLayer(y3)
        return y4

class AutoencoderModel(tf.keras.Model):
    def __init__(
        self,
        input_size,
        **kwargs):
        super(AutoencoderModel, self).__init__(name='autoencoder', **kwargs)
        self.autoencoder = Autoencoder(input_size=input_size)
    
    def call(self, input):
        x_r = self.autoencoder(input)
        #loss = tf.reduce_sum(input_tensor=tf.math.square(tf.norm(tensor=omega - x_r))) # PCA, TODO: add s variable
        #self.add_loss(loss)
        return x_r

def main(argc, argv):
    n_epochs = 1000
    n_classes = 10
    n_samples = 1000 # reduce dataset
    max_val = 255

    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

    # Clear out any prior log data
    # os.system("rm -rf logs")

    logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    file_writer = tf.summary.create_file_writer(logdir)
    #with file_writer.as_default():
    #    images = np.reshape(X_train[0:25], (-1, 28, 28, 1))
    #    tf.compat.v1.summary.image("Training data", images, max_outputs=25)
        
    # file_writer.flush()

    # reduce and normalise data
    X_train = np.array([X.ravel() for X in X_train[:n_samples]]) / max_val
    X_test = np.array([X.ravel() for X in X_test[:n_samples]]) / max_val
    
    input_size = X_train.shape[1]

    # Categorical representation of the labels:
    #y_train, y_test = tf.keras.utils.to_categorical(y_train, num_classes=n_classes), \
    #    tf.keras.utils.to_categorical(y_test, num_classes=n_classes)

    autoencoder = AutoencoderModel(input_size)

    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    autoencoder.compile(optimizer, loss=ReconstructionError())

    # second X_train should be y_train
    autoencoder.fit(X_train, X_train, epochs=n_epochs) # batch_size=n_samples? 

    print("Done")
    return 0

if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))