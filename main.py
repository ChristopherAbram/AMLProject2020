import numpy as np
import sys, os

import tensorflow as tf
from tensorflow.keras import datasets, optimizers
from datetime import datetime
import matplotlib.pyplot as plt

# tf.python.framework.ops.disable_eager_execution()

if(tf.executing_eagerly()):
    print('Eager execution is enabled (running operations immediately)\n')
else:
    print('Eager execution is off\n')

print(('\nYour devices that are available:\n{0}').format(
    [device.name for device in tf.config.experimental.list_physical_devices()]
))

class DeepGAE(tf.Module):
    def __init__(self, input_size, n_classes, file_writer):
        super(DeepGAE, self).__init__(name="DeepGAE")
        self.input_size = input_size
        self.n_classes = n_classes
        self.file_writer = file_writer
        self.parameter_list = None

    def compute_reconstruction_set(self, X):
        pass 
    
    def compute_reconstruction_weights(self, X):
        pass

    @tf.function
    def predict(self, X):
        y1 = self.layer1(X)
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)
        return self.layer4(y3)

    def __call__(self, X):
        return self.predict(X)

    def compile(self, optimizer):
        self.optimizer = optimizer

        # Hidden layers:
        self.W1 = tf.Variable(tf.random.normal(shape=[self.input_size, 200]), name='W1', trainable=True, dtype=tf.float32)
        self.b1 = tf.Variable(tf.random.normal(shape=[200]), name='b1', trainable=True)
        self.layer1 = lambda X: tf.nn.sigmoid(X @ self.W1 + self.b1) # first hidden layer (200 nodes)

        self.W2 = tf.Variable(tf.random.normal(shape=[200, 100]), name='W2', trainable=True)
        self.b2 = tf.Variable(tf.random.normal(shape=[100]), name='b2', trainable=True)
        self.layer2 = lambda X: tf.nn.sigmoid(X @ self.W2 + self.b2) # second hidden layer / bottle neck (compressed) (100 nodes)

        self.W3 = tf.Variable(tf.random.normal(shape=[100, 200]), name='W3', trainable=True)
        self.b3 = tf.Variable(tf.random.normal(shape=[200]), name='b3', trainable=True)
        self.layer3 = lambda X: tf.nn.sigmoid(X @ self.W3 + self.b3) # third hidden layer (200 nodes)

        # Output layer:
        self.W4 = tf.Variable(tf.random.normal(shape=[200, self.input_size]), name='W4', trainable=True, dtype=tf.float32)
        self.b4 = tf.Variable(tf.random.normal(shape=[self.input_size]), name='b4', trainable=True)
        self.layer4 = lambda X: tf.nn.sigmoid(X @ self.W4 + self.b4) # output layer / reconstruction

        self.parameter_list = [
            self.W1, self.b1,
            self.W2, self.b2,
            self.W3, self.b3,
            self.W4, self.b4,
        ]

    def __loss(self, omega, S, X_pred):
        return tf.reduce_mean(S * tf.square(tf.norm(omega - X_pred))) # TODO: might define in subclass

    def __gradients(self, X, omega, S):
        with tf.GradientTape() as tape:
            X_pred = self.predict(X)
            loss_value = self.__loss(omega, S, X_pred)
            grads = tape.gradient(loss_value, self.parameter_list)
        return loss_value, grads

    def fit(self, X, y, epochs=10000): # epoch = expected number of iterations until convergence
        with self.file_writer.as_default():
            omega = self.compute_reconstruction_set(X) # compute the reconstruction set, Ω, based on x_i
            S = self.compute_reconstruction_weights(X) # compute the reconstruction weights, S, based on x_i
            for epoch in range(epochs):
                # Algorithm 1 Iterative learning procedure for Generalized Autoencoder
                loss_value = 0
                for X_batch, y_batch in X:
                    loss_value, grads = self.__gradients(X_batch, omega, S)
                    self.optimizer.apply_gradients(zip(grads, self.parameter_list)) # minimize the reconstruction error using SGD
                
                # TODO:
                y = X.map(lambda X_batch, y_batch: (self.encode(X_batch), y_batch))
                omega = self.compute_reconstruction_set(y) # compute the reconstruction set, Ω, based on y_i
                S = self.compute_reconstruction_weights(y) # compute the reconstruction weights, S, based on y_i
                    
                if epoch % 10 == 0:
                    tf.summary.scalar('loss', loss_value, step=epoch)
                    print("Epoch: ", epoch, "loss_value=", loss_value)

    def evaluate(self):
        pass

class dGAE_PCA(DeepGAE):
    def compute_reconstruction_set(self, X):
        return X

    def compute_reconstruction_weights(self, X):
        return np.ones(X.shape)

def save_model(model, filepath):
    tf.saved_model.save(model, filepath)

def load_model(filepath):
    return tf.saved_model.load(filepath)

def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    x = tf.reshape(x, [28*28])
    y = tf.cast(y, tf.int32)
    return x, y

def postprocess(X, image_shape=(28, 28)):
    return tf.reshape(X, image_shape) * 255

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
    n_epochs = 200
    n_classes = 10
    batch_size = 64
    n_samples = 1000 # reduce dataset
    image_shape = (28, 28)
    load_existing_model = True
    save_path = '.model'

    (X_train, y_train), (X_test, y_test) = dataset(batch_size, n_samples)

    logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    file_writer = tf.summary.create_file_writer(logdir)
    # with file_writer.as_default():
    #     images = np.reshape(X_train[0:25], (-1, 28, 28, 1))
    #     tf.summary.image("Training data", images, max_outputs=25)
        
    input_size = 28 * 28
    
    
    if os.path.exists(save_path) and load_existing_model:
        print("Loading existing model...")
        model = load_model(save_path)
        
    else:
        print("Training new model...")
        
        model = dGAE_PCA(input_size, n_classes, file_writer)
        # Note! We use keras optimizer.
        # TODO: try with momentum..
        model.compile(optimizers.SGD(learning_rate=0.01))
        model.fit(X_train, y_train, epochs=n_epochs)

        print("Saving model...")
        save_model(model, save_path)

    # Reconstruct test samples using model:
    Xr = X_test.unbatch() \
        .map(lambda X, y: (postprocess(X, image_shape), X)) \
        .batch(batch_size) \
        .map(lambda Xp, X: (Xp, model.predict(X))) \
        .unbatch() \
        .map(lambda Xp, X: (Xp, postprocess(X))) \
        .take(15)

    # Visualize reconstructed images:
    fig, axs = plt.subplots(5, 6)
    axs_ = axs.ravel()
    for i, (img_org, img_pred) in enumerate(Xr):
        axs_[i * 2].imshow(img_pred, cmap='gray')
        axs_[i * 2 + 1].imshow(img_org, cmap='gray')
        
    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))