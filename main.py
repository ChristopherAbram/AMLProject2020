import numpy as np
import abc
import sys, os
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import datasets
import logging
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
    **kwargs):
        super(AutoencoderLayer, self).__init__(name="autoencoderlayer", **kwargs)
        self.input_size = input_size
        self.nodes = nodes

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=[self.input_size, self.nodes],
            initializer="random_normal",
            trainable=True,
            name="w"
        )
        self.b = self.add_weight(
            shape=[self.nodes], 
            initializer="random_normal", 
            trainable=True,
            name="b"
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
        self.firstHiddenLayer = AutoencoderLayer(S, input_size, 200) # TODO: create separate encoder and decoder
        self.secondHiddenLayer = AutoencoderLayer(S, 200, 100)
        self.thirdHiddenLayer = AutoencoderLayer(S, 100, 200)
        self.outputLayer = AutoencoderLayer(S, 200, input_size)

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


def main(argc, argv):
    load_model = True
    n_epochs = 10
    n_classes = 10
    n_samples = 1000 # reduce dataset
    max_val = 255

    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

    # Clear out any prior log data
    # os.system("rm -rf logs")

    logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    file_writer = tf.summary.create_file_writer(logdir)

    def log_autoencoder_images(epoch, logs):
        with file_writer.as_default():
            images = np.reshape(X_train[0:25], (-1, 28, 28, 1))
            tf.summary.image("Training data", images, max_outputs=25, step=epoch)

    # Define the per-epoch callback.
    autoencoder_images_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_autoencoder_images)
    
    # reduce and normalise data
    X_train = np.array([X.ravel() for X in X_train[:n_samples]]) / max_val
    X_test = np.array([X.ravel() for X in X_test[:n_samples]]) / max_val
    
    input_size = X_train.shape[1]

    method = PCA()
    omega, S = method.build_reconstruction_set(X_train)

    # Categorical representation of the labels:
    Y_train, y_test = tf.keras.utils.to_categorical(y_train[:n_samples], num_classes=n_classes), \
        tf.keras.utils.to_categorical(y_test[:n_samples], num_classes=n_classes)

    # Build model, show summary:
    autoencoder = AutoencoderModel(input_size, S)
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    autoencoder.compile(optimizer, loss=ReconstructionError())
    autoencoder.build(X_train.shape) # have to pass X_train.shape, but not needed

    autoencoder.summary()
    tf.keras.utils.plot_model(autoencoder, 'img/dgae_pca.png')

    path = '.weight/pca.h5'
    if os.path.exists(path) and load_model:
        autoencoder.load_weights(path)

    else:
        autoencoder.fit(X_train, 
            omega,
            verbose=1,
            callbacks=[tensorboard_callback, autoencoder_images_callback],
            epochs=n_epochs, 
            batch_size=n_samples)
        
        autoencoder.save_weights(path, overwrite=True)

    # Use the model to output images:
    X_reconstructed = autoencoder.predict(X_test[:10])



    print("Done")
    return 0



if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))