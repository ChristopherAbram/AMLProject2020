from datetime import datetime
import tensorflow as tf
from tensorflow.keras import datasets
import logging
import matplotlib.pyplot as plt

logging.getLogger("tensorflow").setLevel(logging.INFO)

class ReconstructionError(tf.keras.losses.Loss):
    def __init__(self):
        super(ReconstructionError, self).__init__(name="reconstructionerror")

    def call(self, omega, x_r):
        return tf.reduce_sum(tf.math.square(tf.norm(omega - x_r))) # PCA, TODO: add s variable


class AutoencoderLayer(tf.keras.layers.Layer):
    def __init__(self, 
    S, # reconstruction weights
    input_size, 
    nodes,
    name="autoencoderlayer",
    **kwargs):
        super(AutoencoderLayer, self).__init__(name, **kwargs)
        self.input_size = input_size
        self.nodes = nodes

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=[self.input_size, self.nodes],
            initializer="random_normal",
            trainable=True,
            name="W:" + self.name
        )
        self.b = self.add_weight(
            shape=[self.nodes], 
            initializer="random_normal", 
            trainable=True,
            name="b:" + self.name
        )

    def call(self, x):
        res = tf.nn.sigmoid(tf.matmul(x, self.w) + self.b)
        print("[INFO] returning autoencoder layer result: ", res)
        return res

class Autoencoder(tf.keras.layers.Layer):
    def __init__(
        self,
        S, # reconstruction weights
        input_size,
        **kwargs):
        super(Autoencoder, self).__init__(name="autoencoder", **kwargs)
        self.firstHiddenLayer = AutoencoderLayer(S, input_size, 200, name="dense1") # TODO: create separate encoder and decoder
        self.secondHiddenLayer = AutoencoderLayer(S, 200, 100, name="dense2")
        self.thirdHiddenLayer = AutoencoderLayer(S, 100, 200, name="dense3")
        self.outputLayer = AutoencoderLayer(S, 200, input_size, name="dense4")

    def call(self, x):
        y = self.firstHiddenLayer(x)
        y = self.secondHiddenLayer(y)
        y = self.thirdHiddenLayer(y)
        y = self.outputLayer(y)
        print("[INFO] returning autoencoder result:", y)
        return y

class AutoencoderModel(tf.keras.Model):
    def __init__(
        self,
        input_size,
        S, # reconstruction weights
        **kwargs):
        super(AutoencoderModel, self).__init__(name='autoencodermodel', **kwargs)
        self.autoencoder = Autoencoder(S=S, input_size=input_size)
    
    def call(self, x):
        x_r = self.autoencoder(x)
        print("[INFO] returning autoencoder model result: ", x_r)
        return x_r


class Methods(abc.ABC):
    @abc.abstractmethod
    def build_reconstruction_set(self, X):
        """
        Builds reconstruction set per each element in input X.
        Returns reconstruction set and the set of weights for x_i
        """
        pass


class PCA(Methods):
    def __init__(self):
        pass

    def build_reconstruction_set(self, X):
        S = np.ones(X.shape[0])
        return X, S


def preprocess(X, y, max_value= 255.0, n_classes=10):
    X_, y_ = X.copy().astype(np.float32), y.copy() # we don't wont to lose original data
    X_ = X_.reshape((X_.shape[0], X_.shape[1] * X_.shape[2])) / max_value
    y_ = tf.keras.utils.to_categorical(y, num_classes=n_classes)
    return X_, y_
