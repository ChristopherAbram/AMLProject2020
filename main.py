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
    def __init__(self, layers, n_classes, file_writer):
        super(DeepGAE, self).__init__(name="DeepGAE")
        self.layers = layers
        self.n_classes = n_classes
        self.file_writer = file_writer
        self.parameter_list = None
        self.encoding_W = []
        self.encoding_b = []
        self.encoding_layers = []
        self.decoding_W = []
        self.decoding_b = []
        self.decoding_layers = []
        self.recalculate_reconstruction_sets = True

    @tf.function
    def predict(self, X):
        return self.decode(self.encode(X))

    @tf.function
    def encode(self, X):
        Y = X
        for layer in self.encoding_layers:
            Y = layer(Y)
        return Y
    
    @tf.function
    def decode(self, Y):
        X = Y
        for layer in self.decoding_layers:
            X = layer(X)
        return X

    @tf.function
    def __call__(self, X):
        return self.predict(X)

    def compile(self, optimizer):
        self.optimizer = optimizer
        self.parameter_list = []

        # Hidden layers:
        layers = self.layers[1:]
        last_layer_size = self.layers[0]
        for i, layer in enumerate(layers):
            self.encoding_W.append(tf.Variable(
                tf.random.normal(shape=[last_layer_size, layer]), name='e-W%d' % i, trainable=True, dtype=tf.float32))
            self.encoding_b.append(tf.Variable(
                tf.random.normal(shape=[layer]), name='e-b%d' % i, trainable=True))
            self.encoding_layers.append(lambda X, j=i: tf.nn.sigmoid(X @ self.encoding_W[j] + self.encoding_b[j]))
            self.parameter_list.append(self.encoding_W[i])
            self.parameter_list.append(self.encoding_b[i])
            last_layer_size = layer

        layers = self.layers[::-1][1:]
        for i, layer in enumerate(layers):
            self.decoding_W.append(tf.Variable(
                tf.random.normal(shape=[last_layer_size, layer]), name='d-W%d' % i, trainable=True, dtype=tf.float32))
            self.decoding_b.append(tf.Variable(
                tf.random.normal(shape=[layer]), name='d-b%d' % i, trainable=True))
            self.decoding_layers.append(lambda X, j=i: tf.nn.sigmoid(X @ self.decoding_W[j] + self.decoding_b[j]))
            self.parameter_list.append(self.decoding_W[i])
            self.parameter_list.append(self.decoding_b[i])
            last_layer_size = layer

        return 

    # Eqn. 3
    def __reconstruction_error_i(self, omega_j, S_j, X_pred):
        # bring X_pred to same dimensions as omega_j to allow for subtraction. 
        # Will enable us to subtract the same data point from each reconstuction element
        X_pred_dim = tf.map_fn(
            fn=lambda x: tf.convert_to_tensor([x] * omega_j.shape[1]), 
            elems=X_pred, 
            fn_output_signature=tf.TensorSpec(shape=(omega_j.shape[1], omega_j.shape[2])))
            
        return tf.reduce_sum(S_j * tf.square(tf.norm(omega_j - X_pred_dim, axis=2)), axis=1)

    # Eqn. 4
    def __loss(self, omega, S, X_pred):
        # return tf.reduce_sum(S * tf.square(tf.norm(omega - X_pred)))
        return tf.reduce_sum(self.__reconstruction_error_i(omega, S, X_pred), axis=None)

    def __gradients(self, X, omega, S):
        with tf.GradientTape() as tape:
            X_pred = self.predict(X)
            loss_value = self.__loss(omega, S, X_pred)
            grads = tape.gradient(loss_value, self.parameter_list)
        return loss_value, grads

    def fit(self, X, y, epochs=10000): # epoch = expected number of iterations until convergence
        with self.file_writer.as_default():

            # 1. Compute the reconstruction weights Si from {x_i} and
            #    determine the reconstruction set i, e.g. by k-nearest neighbor
            X = X.map(lambda X_batch, y_batch: (
                X_batch, 
                y_batch, 
                self.compute_reconstruction_set(X_batch, y_batch), 
                self.compute_reconstruction_weights(X_batch, y_batch)
            ))

            for epoch in range(epochs):
                # 2. Minimize E in Eqn.4 using the stochastic gradient 
                #    descent and update theta for t steps.
                loss_value = 0
                for X_batch, y_batch, omega_batch, S_batch in X:

                    # FIXME: For debugging:
                    # om_ = self.compute_reconstruction_set(X_batch, y_batch) 
                    # s_ = self.compute_reconstruction_weights(X_batch, y_batch)
                    # re_ = self.__reconstruction_error_i(om_, s_, self.predict(X_batch))

                    loss_value, grads = self.__gradients(X_batch, omega_batch, S_batch)
                    # minimize the reconstruction error using SGD
                    self.optimizer.apply_gradients(zip(grads, self.parameter_list)) 
                
                # 3. Compute the hidden representation y_i, and update S_i and omega_i from y_i
                if self.recalculate_reconstruction_sets:
                    X = X.map(self.recalculate_reconstruction)
                    
                if epoch % 10 == 0:
                    tf.summary.scalar('loss', loss_value, step=epoch)
                    print("Epoch: ", epoch, "loss_value=", loss_value)

    def recalculate_reconstruction(self, X_batch, y_batch, omega_batch, s_batch):
        # omega and s are discarded and recalculated
        encoded = self.encode(X_batch)
        return (
            X_batch, 
            y_batch, 
            self.compute_reconstruction_set(encoded, y_batch), 
            self.compute_reconstruction_weights(encoded, y_batch)
        )

    def evaluate(self):
        pass

class dGAE_PCA(DeepGAE):
    def __init__(self, layers, n_classes, file_writer):
        super(dGAE_PCA, self).__init__(layers, n_classes, file_writer)
        self.recalculate_reconstruction_sets = True

    def compute_reconstruction_set(self, X_batch, y_batch):
        return tf.map_fn(fn=lambda x: tf.convert_to_tensor([x,x]), elems=X_batch, 
            fn_output_signature=tf.TensorSpec(shape=(2, X_batch.shape[1])))

    def compute_reconstruction_weights(self, X_batch, y_batch):
        return tf.map_fn(fn=lambda x: tf.convert_to_tensor([0.5,0.5], dtype=tf.float32), elems=X_batch, 
            fn_output_signature=tf.TensorSpec(shape=(2,)))

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
    n_epochs = 100
    n_classes = 10
    batch_size = 64
    n_samples = 1000 # reduce dataset
    image_shape = (28, 28)
    load_existing_model = False
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
        
        model = dGAE_PCA([input_size, 200, 100], n_classes, file_writer)
        # Note! We use keras optimizer.
        # TODO: try with momentum..
        model.recalculate_reconstruction_sets = True
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