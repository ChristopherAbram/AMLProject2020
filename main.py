import numpy as np
import sys, os, math

import tensorflow as tf
from tensorflow.keras import datasets, optimizers, layers
from datetime import datetime
import matplotlib.pyplot as plt
import functools as ft
import math

# tf.python.framework.ops.disable_eager_execution()
tf.config.run_functions_eagerly(False)

print('Intra op threads: {}'.format(tf.config.threading.get_intra_op_parallelism_threads()))
print('Inter op threads: {}'.format(tf.config.threading.get_inter_op_parallelism_threads()))

if(tf.executing_eagerly()):
    print('Eager execution is enabled (running operations immediately)\n')
else:
    print('Eager execution is off\n')

print(('\nYour devices that are available:\n{0}').format(
    [device.name for device in tf.config.experimental.list_physical_devices()]
))

# From: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {iteration}/{total} {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

@tf.function
def knn(xj, data, k=18):
    distances = tf.norm(data - xj, axis=1)
    _, top_k_indices = tf.nn.top_k(tf.negative(distances), k=k)
    return top_k_indices

class DeepGAE(tf.Module):
    def __init__(self, file_writer):
        super(DeepGAE, self).__init__(name="DeepGAE")
        self.layers = None
        self.n_classes = None
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
    def predict(self, X, training=False):
        return self.decode(self.encode(X))

    @tf.function
    def encode(self, X, training=False):
        Y = X
        for layer in self.encoding_layers:
            Y = layer(Y)
        return Y
    
    @tf.function
    def decode(self, Y, training=False):
        X = Y
        for layer in self.decoding_layers:
            X = layer(X)
        return X

    @tf.function
    def __call__(self, X):
        return self.predict(X)

    def compile(self, optimizer, layers, n_classes):
        self.layers = layers
        self.n_classes = n_classes
        self.optimizer = optimizer
        self.parameter_list = []

        initializer = tf.initializers.glorot_uniform()
        zero_init = tf.initializers.zeros()
        # Hidden layers:
        layers = self.layers[1:]
        last_layer_size = self.layers[0]
        for i, layer in enumerate(layers):
            self.encoding_W.append(tf.Variable(
                initializer(shape=[last_layer_size, layer]), name='e-W%d' % i, trainable=True, dtype=tf.float32))
            self.encoding_b.append(tf.Variable(
                zero_init(shape=[layer]), name='e-b%d' % i, trainable=True))
            self.encoding_layers.append(lambda X, j=i: tf.nn.sigmoid(X @ self.encoding_W[j] + self.encoding_b[j]))
            self.parameter_list.append(self.encoding_W[i])
            self.parameter_list.append(self.encoding_b[i])
            last_layer_size = layer

        layers = self.layers[::-1][1:]
        for i, layer in enumerate(layers):
            self.decoding_W.append(tf.Variable(
                initializer(shape=[last_layer_size, layer]), name='d-W%d' % i, trainable=True, dtype=tf.float32))
            self.decoding_b.append(tf.Variable(
                zero_init(shape=[layer]), name='d-b%d' % i, trainable=True))
            self.decoding_layers.append(lambda X, j=i: tf.nn.sigmoid(X @ self.decoding_W[j] + self.decoding_b[j]))
            self.parameter_list.append(self.decoding_W[i])
            self.parameter_list.append(self.decoding_b[i])
            last_layer_size = layer

    # Eqn. 4
    @tf.function
    def loss(self, omega, S, X_pred):
        return tf.reduce_sum(
                    S * tf.square(tf.norm(
                        omega - tf.expand_dims(X_pred, axis=1), axis=2)))

    def preprocess(self, X, y, batch_size=64):
        y_ = tf.one_hot(y, depth=self.n_classes)
        return tf.data.Dataset.from_tensor_slices((X, y_)) \
            .shuffle(10000) \
            .batch(batch_size) \
            .map(lambda X_batch, y_batch: (X_batch, y_batch, None))

    def recompute(self, data):
        return data

    def fit(self, X, y, epochs=10000, pretrain_epochs=1, batch_size=64, initial_epoch=1, steps_per_epoch=None): 
        # epoch = expected number of iterations until convergence
        with self.file_writer.as_default():
            # 1. Compute the reconstruction weights Si from {x_i} and
            #    determine the reconstruction set i, e.g. by k-nearest neighbor
            data = self.preprocess(X, y, batch_size)

            for epoch in range(initial_epoch, epochs + 1):
                # 2. Minimize E in Eqn.4 using the stochastic gradient 
                #    descent and update theta for t steps.
                loss_value = 0.
                num_steps_ = X.shape[0] // batch_size
                printProgressBar(0, num_steps_, 
                    prefix=f'Epochs {epoch}/{epochs}', suffix='', length=50)

                for step, (X_batch, y_batch, encoded_batch) in enumerate(data):
                    omega_batch, S_batch = self.compute_reconstruction(
                        encoded_batch if epoch > pretrain_epochs else None, X_batch, y_batch)
                    
                    with tf.GradientTape() as tape:
                        X_pred = self.predict(X_batch, training=True)
                        loss_value = self.loss(omega_batch, S_batch, X_pred)
                    grads = tape.gradient(loss_value, self.parameter_list)
                    self.optimizer.apply_gradients(zip(grads, self.parameter_list)) 

                    printProgressBar(step + 1, num_steps_, 
                        prefix=f'Epochs {epoch}/{epochs}', 
                        suffix=f'loss= {loss_value.numpy()}', length=50)

                    if steps_per_epoch is not None and step > steps_per_epoch:
                        break

                tf.summary.scalar('loss', loss_value, step=epoch)
                
                # 3. Compute the hidden representation y_i, and update S_i and omega_i from y_i
                if self.recalculate_reconstruction_sets:
                    data = self.recompute(data)

    @tf.function
    def get_class_label(self, y):
        return tf.argmax(y, axis=0)

    def get_class_division(self, X, y, equal_sizes=True):
        if not equal_sizes:
            classes, sizes = [], []
            for Ci in range(self.n_classes):
                inx = np.where(y==Ci)
                sizes.append(inx[0].size)
                classes.append(X[inx].tolist())
                classes = tf.ragged.constant(classes, ragged_rank=1)
            return classes, sizes
        else:
            classes, sizes = [], []
            for Ci in range(self.n_classes):
                classes.append(X[np.where(y==Ci)])
            sizes = [cl.shape[0] for cl in classes]
            min_size = min(sizes)
            classes =  np.array([cl[:min_size,:] for cl in classes])
            classes = tf.convert_to_tensor(classes)
            sizes = [cl.shape[0] for cl in classes]
            return classes, sizes

    def save(self, filepath):
        tf.saved_model.save(self, filepath)

    @classmethod
    def load_model(self, filepath):
        return tf.saved_model.load(filepath)

    def evaluate(self):
        pass

class DeepPCA(DeepGAE):
    def __init__(self, file_writer):
        super(DeepPCA, self).__init__(file_writer)
        self.recalculate_reconstruction_sets = False

    @tf.function
    def compute_reconstruction(self, encoded_batch, X_batch, y_batch):
        omega = tf.map_fn(fn=lambda x: tf.convert_to_tensor([x]), elems=X_batch, 
            fn_output_signature=tf.TensorSpec(shape=(1, X_batch.shape[1])))
        S = tf.map_fn(fn=lambda x: tf.convert_to_tensor([1.0], dtype=tf.float32), elems=X_batch, 
            fn_output_signature=tf.TensorSpec(shape=(1,)))
        return omega, S

class DeepLDABalanced(DeepGAE):
    def __init__(self, file_writer):
        super(DeepLDABalanced, self).__init__(file_writer)
        self.recalculate_reconstruction_sets = False
        self.omega_classes = None

    def preprocess(self, X, y, batch_size=64):
        self.omega_classes, self.omega_counts = self.get_class_division(X, y)
        return super(DeepLDABalanced, self).preprocess(X, y, batch_size)
    
    @tf.function
    def compute_reconstruction(self, encoded_batch, X_batch, y_batch):
        omega = tf.map_fn(
            fn=lambda y: tf.gather(self.omega_classes, self.get_class_label(y)),
            elems=y_batch,
            fn_output_signature=tf.TensorSpec(shape=[self.omega_counts[0], X_batch.shape[1]]))
        S = tf.map_fn(
            fn=lambda y: tf.convert_to_tensor([1. / self.omega_counts[0]]),
            elems=y_batch, 
            fn_output_signature=tf.TensorSpec(shape=(1,)))
        return omega, S
            
class DeepLDA(DeepGAE):
    def __init__(self, file_writer):
        super(DeepLDA, self).__init__(file_writer)
        self.recalculate_reconstruction_sets = False
        self.omega_classes = None

    def preprocess(self, X, y, batch_size):
        data = super(DeepLDA, self).preprocess(X, y, batch_size)
        self.omega_classes = self.get_class_division(data)
        return data

    def get_class_division(self, data):
        classes = [[] for i in range(self.n_classes)]
        for X_batch, y_batch, encoded_batch in data:
            for i, x in enumerate(X_batch):
                class_label = self.get_class_label(y_batch[i])
                classes[class_label].append(x)
        
        classes = [tf.RaggedTensor.from_tensor(c) for c in classes]
        return classes

    @tf.function
    def loss(self, omega, S, X_pred):
        s = 0
        for i, omega_i in enumerate(omega):
            s += tf.reduce_sum(
                    S[i] * tf.square(tf.norm(
                        omega_i.to_tensor() - X_pred[i], axis=1)))
        return s
    
    @tf.function
    def compute_reconstruction(self, encoded_batch, X_batch, y_batch):
        omega = tf.map_fn(
            fn=lambda i: self.omega_classes[
                self.get_class_label(y_batch[i.numpy()])
            ],
            elems=tf.range(X_batch.shape[0]),
            dtype=X_batch.dtype,
            fn_output_signature=tf.RaggedTensorSpec(shape=[None, X_batch.shape[1]]))

        S = tf.map_fn(
            fn=lambda i: tf.convert_to_tensor(
                [1. / self.omega_classes[
                    self.get_class_label(y_batch[i.numpy()])].shape[0]], 
                dtype=tf.float32
            ), 
            elems=tf.range(X_batch.shape[0]), 
            fn_output_signature=tf.RaggedTensorSpec(shape=(1,)))

        return omega, S

class DeepLE(DeepGAE):
    def __init__(self, file_writer, k=10):
        super(DeepLE, self).__init__(file_writer)
        self.recalculate_reconstruction_sets = True
        self.k = k
        self.t = 1.
        self.data = None
        self.data_encoded = None

    def preprocess(self, X, y, batch_size=64):
        self.data = tf.convert_to_tensor(X)
        return super(DeepLE, self).preprocess(X, y, batch_size)

    def recompute(self, data):
        data = data.map(lambda X_batch, y_batch, encoded_batch: (X_batch, y_batch, self.encode(X_batch)))
        self.data_encoded = self.encode(self.data)
        return data

    @tf.function
    def compute_reconstruction(self, encoded_batch, X_batch, y_batch):
        X_batch_ = X_batch if encoded_batch is None else encoded_batch
        data_ = self.data if encoded_batch is None else self.data_encoded
        omega = tf.map_fn(
            fn=lambda x: tf.gather(self.data, knn(x, data_, k=self.k)),
            elems=X_batch_, 
            fn_output_signature=tf.TensorSpec(shape=(self.k, X_batch.shape[1])))
        S = tf.exp(tf.negative(
                tf.square(tf.norm(
                    tf.expand_dims(X_batch, axis=1) - omega, axis=2))) / self.t)
        return omega, S


class DeepMFA(DeepGAE):
    def __init__(self, file_writer, k=18):
        super(DeepMFA, self).__init__(file_writer)
        self.recalculate_reconstruction_sets = True
        self.omega_classes = None
        self.k = k

    def get_encoded_class_division(self, data):
        classes = [[] for i in range(self.n_classes)]
        for X_batch, y_batch, encoded_batch in data:
            for i, x in enumerate(encoded_batch):
                class_label = self.get_class_label(y_batch[i])
                classes[class_label].append(x)
        
        classes = [tf.RaggedTensor.from_tensor(c) for c in classes]
        return classes

    def preprocess(self, X, y):
        X = X.map(lambda X_batch, y_batch: (X_batch, y_batch, None))

        # 1. group points into classes
        self.omega_classes = np.array(self.get_class_division(X, y))
        # 2. 
        self.omega_classes_complement = []
        for Ci, X_class in enumerate(self.omega_classes):
            class_inx=list(range(self.n_classes))
            class_inx.pop(Ci)

            self.omega_classes_complement.append(
                tf.concat(self.omega_classes[class_inx].tolist(), axis=0).to_tensor())

        return X, y

    def recompute(self, X, y):
        X = X.map(lambda X_batch, y_batch, encoded_batch: (X_batch, y_batch, self.encode(X_batch)))

        self.omega_encoded_classes = np.array(self.get_encoded_class_division(X, y))
        self.omega_encoded_classes_complement = []
        for Ci, X_class in enumerate(self.omega_encoded_classes):
            class_inx=list(range(self.n_classes))
            class_inx.pop(Ci)
            self.omega_encoded_classes_complement.append(
                tf.concat(self.omega_encoded_classes[class_inx].tolist(), axis=0).to_tensor())

        return X, y
    
    def compute_reconstruction_set(self, encoded_batch, X_batch, y_batch):
        X_batch_ = X_batch
        omega_classes_ = self.omega_classes
        omega_classes_complement_ = self.omega_classes_complement
        if encoded_batch is not None:
            X_batch_ = encoded_batch
            omega_classes_ = self.omega_encoded_classes
            omega_classes_complement_ = self.omega_encoded_classes_complement

        def mappingFunc(i):
            Ci = self.get_class_label(y_batch[i.numpy()])
            omega_k1_inx = knn(X_batch_[i], omega_classes_[Ci].to_tensor())
            omega_k2_inx = knn(X_batch_[i], omega_classes_complement_[Ci])
            return tf.concat([
                tf.gather(self.omega_classes[Ci].to_tensor(), omega_k1_inx), 
                tf.gather(self.omega_classes_complement[Ci], omega_k2_inx)], axis=0)

        return tf.map_fn(
            fn=lambda i: mappingFunc(i),
            elems=tf.range(X_batch.shape[0]), 
            fn_output_signature=tf.TensorSpec(shape=(2*self.k, X_batch.shape[1])))

    def compute_reconstruction_weights(self, encoded_batch, X_batch, y_batch):
        return tf.map_fn(
            fn=lambda i: \
                tf.convert_to_tensor(
                    ([1.0] * self.k) + ([-1.0] * self.k), 
                dtype=tf.float32), 
            elems=tf.range(X_batch.shape[0]), 
            fn_output_signature=tf.TensorSpec(shape=(2*self.k,)))

def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    x = tf.reshape(x, [28*28])
    y = tf.cast(y, tf.int32)
    return x, y

def postprocess(X, image_shape=(28, 28)):
    return tf.reshape(X, image_shape) * 255

def get_class_label(y): #possibly shouldn't be member of class
    return tf.argmax(y, axis=0).numpy()

def factory(model_name):
    logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    file_writer = tf.summary.create_file_writer(logdir)
    if model_name == "PCA":
        return DeepPCA(file_writer)
    elif model_name == "LDABalanced":
        return DeepLDABalanced(file_writer)
    elif model_name == "LDA":
        return DeepLDA(file_writer)
    elif model_name == "MFA":
        return DeepMFA(file_writer)
    elif model_name == "LE":
        return DeepLE(file_writer)


MNIST_size = 60000

def dataset(num_sets = 30, validation_ratio = 0.2):
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

    # Normalize and reshape:
    image_shape = X_train[0].shape
    X_train = (X_train.astype(np.float32) / 255.0).reshape((X_train.shape[0], image_shape[0] * image_shape[1]))
    X_test = (X_train.astype(np.float32) / 255.0).reshape((X_train.shape[0], image_shape[0] * image_shape[1]))

    # Calculate set sizes
    set_size = math.floor(MNIST_size/num_sets)
    train_size = math.floor(set_size * (1 - validation_ratio))
    valid_size = math.floor(set_size * validation_ratio)

    # Split into training and validation
    train_valid = []
    for i in range(num_sets):
        _X_train = X_train[i*set_size:i*set_size+train_size]
        _y_train = y_train[i*set_size:i*set_size+train_size]
        _X_valid = X_train[i*set_size+train_size:i*set_size+train_size+valid_size]
        _y_valid = y_train[i*set_size+train_size:i*set_size+train_size+valid_size]
        train_valid.append(((_X_train,_y_train),(_X_valid,_y_valid)))

    return train_valid, (X_test, y_test)

def _dataset(X, y, batch_size=64):
    y = tf.one_hot(y, depth=10)
    return tf.data.Dataset.from_tensor_slices((X, y)) \
        .map(preprocess) \
        .shuffle(1000) \
        .batch(batch_size)

def main(argc, argv):
    image_shape = (28, 28)
    load_existing_model = False
    input_size = image_shape[0] * image_shape[1]
    n_classes = 10
    batch_size = 64
    n_samples = 2000

    epochs = [10]
    layers_configs = [[input_size, 200, 30]]
    learning_rates = [0.001]
    momentums = [0.5]
    model_names = ['PCA']
    pretrain_epoch_ratio = 0.0
    
    train_valid_set, test_set = dataset(MNIST_size // n_samples)

    with tf.device('CPU:0'):
        i = 0
        for n_epochs in epochs:
            for layers in layers_configs:
                for learning_rate in learning_rates:
                    for momentum in momentums:
                        for model_name in model_names:
                            layer_string = '-'.join(str(x) for x in layers)
                            print("########################################################")
                            print("Validating config: %s model - %d epochs - %d samples - %s layers - %f learning rate - %f momentum - %d batch size" % (model_name, n_epochs, n_samples, layer_string, learning_rate, momentum, batch_size))
                            (X_train,y_train),(X_valid,y_valid) = train_valid_set[i]

                            save_path = ".models/" + "_".join((model_name,str(n_epochs),str(batch_size),str(n_samples),str(layer_string),str(learning_rate),str(momentum)))

                            if os.path.exists(save_path) and load_existing_model:
                                print("Loading existing model...")
                                model = DeepGAE.load_model(save_path)
                                
                            else:
                                print("Training new model...")
                                
                                model = factory(model_name)
                                # Note! We use keras optimizer.
                                model.compile(optimizers.SGD(learning_rate=learning_rate, momentum=momentum), layers, n_classes)
                                model.fit(X_train, y_train, 
                                    epochs=n_epochs, 
                                    pretrain_epochs=(pretrain_epoch_ratio * n_epochs))
                                print("Saving model...")
                                
                                model.save(save_path)
                                if not os.path.exists(save_path + "/img"):
                                    os.mkdir(save_path + "/img")


                            # ERROR RATE and IMPURITY
                            er_k = 18 # choose k + 1 as the instance itself will also be there.
                        
                            X_valid_encoded = model.encode(X_valid)
                            X_valid_predicted = model.predict(X_valid)

                            errors = 0
                            impurities = 0
                            for i, x_enc in enumerate(X_valid_encoded):
                                top_k_indices = knn(x_enc, X_valid_encoded,er_k)
                                label = y_valid[i]
                                votes_against = 0
                                for index in top_k_indices:
                                    if label != y_valid[index]:
                                        votes_against += 1
                                if votes_against > math.ceil(er_k/2):
                                    errors += 1
                                impurities += votes_against
                            error_rate = errors*100 / n_samples
                            impurity = impurities / (n_samples * er_k)
                            print("ERROR RATE: %f%%" % error_rate)
                            print("AVG. IMPURITY: %f" % impurity)
                            info_file = open(save_path+"/info.txt","w+")
                            info_file.write("ERROR RATE: %f%%" % error_rate)
                            info_file.write("AVG. IMPURITY: %f" % impurity)
                            info_file.close()

                            width = math.sqrt(layers[len(layers) - 1])
                            height = width
                            if not width.is_integer():
                                width = layers[len(layers) - 1]
                                height = 1
                            else:
                                width = int(width)
                                height = int(height)

                            # 2D GRAPH
                            if width == 2:
                                fig_, ax = plt.subplots()
                                class_points_x = [[] for i in range(10)]
                                class_points_y = [[] for i in range(10)]
                                for i, e in enumerate(X_valid_encoded):
                                    pp_e = postprocess(e, (1, 2))
                                    y = y_valid[i]
                                    coord = pp_e.numpy().ravel()
                                    class_points_x[y].append(coord[0])
                                    class_points_y[y].append(coord[1])

                                for label in range(n_classes):
                                    ax.scatter(class_points_x[label], class_points_y[label], label="%d" % label)

                                plt.legend()
                                plt.savefig(save_path + "/img/2d_encode.png")

                            # Visualize reconstructed images:
                            plot_col, plot_row = 3, 9
                            X_valid = X_valid[:plot_col * plot_row]
                            X_valid_encoded = X_valid_encoded[:plot_col * plot_row]
                            X_valid_predicted = X_valid_predicted[:plot_col * plot_row]

                            fig, axs = plt.subplots(plot_row, plot_col * 3, figsize=(10, 10))
                            axs_ = axs.ravel()
                            for i, (x, xp, xen) in enumerate(zip(X_valid, X_valid_predicted, X_valid_encoded)):
                                img_org = postprocess(x, image_shape)
                                img_pred = postprocess(xp, image_shape)
                                img_encoded = postprocess(xen, (height, width))
                                axs_[i * 3].axis('off')
                                axs_[i * 3 + 1].axis('off')
                                axs_[i * 3 + 2].axis('off')
                                axs_[i * 3].imshow(img_org, cmap='gray')
                                axs_[i * 3 + 1].imshow(img_pred, cmap='gray')
                                axs_[i * 3 + 2].imshow(img_encoded, cmap='gray')

                            plt.savefig(save_path + "/img/predict.png")

    return 0


if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))