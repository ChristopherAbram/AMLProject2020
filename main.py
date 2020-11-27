import numpy as np
import sys, os

import tensorflow as tf
from tensorflow.keras import datasets, optimizers
from datetime import datetime
import matplotlib.pyplot as plt
import functools as ft

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

    # Eqn. 4
    def __loss(self, omega, S, X_pred):
        s = 0
        for i, omega_i in enumerate(omega):
            s += tf.reduce_sum(
                    S.to_tensor()[i] * tf.square(tf.norm(
                        omega_i.to_tensor() - X_pred[i], axis=1)))

        # return tf.reduce_sum(
        #             S * tf.square(tf.norm(
        #                 omega - tf.expand_dims(X_pred, axis=1), axis=2))) # works due to broadcasting
        return s

    def __gradients(self, X, omega, S):
        with tf.GradientTape() as tape:
            X_pred = self.predict(X)
            loss_value = self.__loss(omega, S, X_pred)
            grads = tape.gradient(loss_value, self.parameter_list)
        return loss_value, grads

    def preprocess(self, X, y):
        return X, y

    def fit(self, X, y, epochs=10000): 
        # epoch = expected number of iterations until convergence
        with self.file_writer.as_default():
            # 1. Compute the reconstruction weights Si from {x_i} and
            #    determine the reconstruction set i, e.g. by k-nearest neighbor
            self.preprocess(X, y)

            # For debugging:
            # for X_batch, y_batch in X:
            #     encoded = self.encode(X_batch)
            #     rs = self.compute_reconstruction_set(None, X_batch, y_batch),
            #     if rs is None:
            #         kk=0

            #     rw = self.compute_reconstruction_weights(None, X_batch, y_batch)

            # X = X.map(lambda X_batch, y_batch: tf.py_function(
            #     lambda X_batch, y_batch: (
            #         X_batch, 
            #         y_batch,
            #         self.compute_reconstruction_set(None, X_batch, y_batch),
            #         self.compute_reconstruction_weights(None, X_batch, y_batch)
            #     ), [X_batch, y_batch], Tout=[tf.float32, tf.int32, tf.float32, tf.float32]))

            X = X.map(lambda X_batch, y_batch: (X_batch, y_batch, None))

            for epoch in range(epochs):
                # 2. Minimize E in Eqn.4 using the stochastic gradient 
                #    descent and update theta for t steps.
                loss_value = 0
                for X_batch, y_batch, encoded_batch in X:
                    omega_batch = self.compute_reconstruction_set(encoded_batch, X_batch, y_batch)
                    S_batch = self.compute_reconstruction_weights(encoded_batch, X_batch, y_batch)
                    
                    loss_value, grads = self.__gradients(X_batch, omega_batch, S_batch)
                    # minimize the reconstruction error using SGD
                    self.optimizer.apply_gradients(zip(grads, self.parameter_list)) 
                
                # 3. Compute the hidden representation y_i, and update S_i and omega_i from y_i
                # TODO: figure out encoded values
                if self.recalculate_reconstruction_sets:
                    X = X.map(lambda X_batch, y_batch: (X_batch, y_batch, self.encode(X_batch)))
                    # X = X.map(self.recalculate_reconstruction)
                    
                if epoch % 10 == 0:
                    tf.summary.scalar('loss', loss_value, step=epoch)
                    print("Epoch: ", epoch, "loss_value=", loss_value)

    def recalculate_reconstruction(self, X_batch, y_batch, omega_batch, s_batch):
        # omega and s are discarded and recalculated
        encoded = self.encode(X_batch)

        # For debugging:
        # self.compute_reconstruction_set(encoded, X_batch, y_batch), 
        # self.compute_reconstruction_weights(encoded, X_batch, y_batch)

        return (
            X_batch, 
            y_batch, 
            self.compute_reconstruction_set(encoded, X_batch, y_batch), 
            self.compute_reconstruction_weights(encoded, X_batch, y_batch)
        )

    def get_class_label(self, y): #possibly shouldn't be member of class
        return tf.argmax(y, axis=0).numpy()

    def get_class_division(self, X, y): #possibly shouldn't be member of class
        classes = [[] for i in range(self.n_classes)]
        for X_batch, y_batch in X:
            for i, x in enumerate(X_batch):
                class_label = self.get_class_label(y_batch[i])
                classes[class_label].append(x)
        
        classes = [tf.RaggedTensor.from_tensor(c) for c in classes]
        return classes

    def evaluate(self):
        pass

class DeepPCA(DeepGAE):
    def __init__(self, layers, n_classes, file_writer):
        super(DeepPCA, self).__init__(layers, n_classes, file_writer)
        self.recalculate_reconstruction_sets = False

    def compute_reconstruction_set(self, encoded_batch, X_batch, y_batch):
        return tf.map_fn(fn=lambda x: tf.convert_to_tensor([x]), elems=X_batch, 
            fn_output_signature=tf.TensorSpec(shape=(1, X_batch.shape[1])))

    def compute_reconstruction_weights(self, encoded_batch, X_batch, y_batch):
        return tf.map_fn(fn=lambda x: tf.convert_to_tensor([1.0], dtype=tf.float32), elems=X_batch, 
            fn_output_signature=tf.TensorSpec(shape=(1,)))

class DeepLDA(DeepGAE):
    def __init__(self, layers, n_classes, file_writer):
        super(DeepLDA, self).__init__(layers, n_classes, file_writer)
        self.recalculate_reconstruction_sets = False
        self.omega_classes = None

    def preprocess(self, X, y):
        self.omega_classes = self.get_class_division(X, y)
        return X, y
    
    def compute_reconstruction_set(self, encoded_batch, X_batch, y_batch):
        return tf.map_fn(
            fn=lambda i: \
                # Get all instances x of class Ci
                self.omega_classes[
                    self.get_class_label(y_batch[i.numpy()])],
            elems=tf.range(X_batch.shape[0]),
            dtype=X_batch.dtype,
            fn_output_signature=tf.RaggedTensorSpec(shape=[None, X_batch.shape[1]]))

    def compute_reconstruction_weights(self, encoded_batch, X_batch, y_batch):
        return tf.map_fn(
            fn=lambda i: \
                tf.convert_to_tensor(
                    # Compute weight which is inversely proportional to class size
                    [1. / self.omega_classes[
                        self.get_class_label(y_batch[i.numpy()])].shape[0]], 
                    dtype=tf.float32
                ), 
            elems=tf.range(X_batch.shape[0]), 
            fn_output_signature=tf.RaggedTensorSpec(shape=(1,)))

class DeepMFA(DeepGAE):
    def __init__(self, layers, n_classes, file_writer):
        super(DeepMFA, self).__init__(layers, n_classes, file_writer)
        self.recalculate_reconstruction_sets = True
        self.omega_classes = None

    def preprocess(self, X, y):
        # 1. group points into classes
        self.omega_classes = np.array(self.get_class_division(X, y))
        # 2. 

        return X, y
    
    def knn(self, xj, data, k=18):
        distances = tf.norm(xj - data, axis=1)
        _, top_k_indices = tf.nn.top_k(tf.negative(distances), k=k)
        return tf.gather(data, top_k_indices)
        
    
    def compute_reconstruction_set(self, encoded_batch, X_batch, y_batch):
        for Ci, X_class in enumerate(self.omega_classes):
            class_inx=list(range(self.n_classes))
            class_inx.pop(Ci)
            for x in X_class:
                omega_k1 = self.knn(x, tf.convert_to_tensor(X_class))
                omega_k2 = self.knn(x, tf.convert_to_tensor(self.omega_classes[class_inx]))
                # merge the two omegas and put into tensor
        
        return tf.map_fn(fn=lambda x: tf.convert_to_tensor([x]), elems=X_batch, 
            fn_output_signature=tf.TensorSpec(shape=(1, X_batch.shape[1])))

    def compute_reconstruction_weights(self, encoded_batch, X_batch, y_batch):
        return tf.map_fn(fn=lambda x: tf.convert_to_tensor([1.0], dtype=tf.float32), elems=X_batch, 
            fn_output_signature=tf.TensorSpec(shape=(1,)))

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
        
        model = DeepLDA([input_size, 200, 100], n_classes, file_writer)
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
        .map(lambda Xp, X: (Xp, model.predict(X), model.encode(X))) \
        .unbatch() \
        .map(lambda Xp, X, Xen: (Xp, postprocess(X), postprocess(Xen, (10, 10)))) \
        .take(27)

    # Visualize reconstructed images:
    fig, axs = plt.subplots(9, 9, figsize=(10, 10))
    axs_ = axs.ravel()
    for i, (img_org, img_pred, img_encoded) in enumerate(Xr):
        axs_[i * 3].axis('off')
        axs_[i * 3 + 1].axis('off')
        axs_[i * 3 + 2].axis('off')
        axs_[i * 3].imshow(img_pred, cmap='gray')
        axs_[i * 3 + 1].imshow(img_org, cmap='gray')
        axs_[i * 3 + 2].imshow(img_encoded, cmap='gray')

    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))