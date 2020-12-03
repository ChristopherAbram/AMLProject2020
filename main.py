import numpy as np
import sys, os

import tensorflow as tf
from tensorflow.keras import datasets, optimizers, layers
from datetime import datetime
import matplotlib.pyplot as plt
import functools as ft
import math

# tf.python.framework.ops.disable_eager_execution()
# tf.config.run_functions_eagerly(False)
# tf.compat.v1.data.get_output_shapes

# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(100)

# os.environ['TF_NUM_INTEROP_THREADS'] = '100'
# os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

# tf.device('CPU:0')


print('Intra op threads: {}'.format(tf.config.threading.get_intra_op_parallelism_threads()))
print('Inter op threads: {}'.format(tf.config.threading.get_inter_op_parallelism_threads()))

if(tf.executing_eagerly()):
    print('Eager execution is enabled (running operations immediately)\n')
else:
    print('Eager execution is off\n')

print(('\nYour devices that are available:\n{0}').format(
    [device.name for device in tf.config.experimental.list_physical_devices()]
))

class DeepGAE(tf.Module):
    def __init__(self, file_writer):
        super(DeepGAE, self).__init__(name="DeepGAE")
        self.model_name = "GAE"
        self.layers = []
        self.n_classes = 0
        self.file_writer = file_writer
        self.parameter_list = None
        self.encoding_W = []
        self.encoding_b = []
        self.encoding_layers = []
        self.decoding_W = []
        self.decoding_b = []
        self.decoding_layers = []
        self.recalculate_reconstruction_sets = True

        self.encoder = None
        self.decoder = None
        self.autoencoder = None

    @tf.function
    def predict(self, X, training=False):
        # return self.autoencoder(X, training=training)
        return self.decode(self.encode(X))

    @tf.function
    def encode(self, X, training=False):
        # return self.encoder(X, training=training)
        Y = X
        for layer in self.encoding_layers:
            Y = layer(Y)
        return Y
    
    @tf.function
    def decode(self, Y, training=False):
        # return self.decoder(Y, training=training)
        X = Y
        for layer in self.decoding_layers:
            X = layer(X)
        return X

    @tf.function
    def __call__(self, X):
        return self.predict(X)

    def compile(self, optimizer, layers, n_classes):
        self.n_classes = n_classes
        self.optimizer = optimizer
        self.layers = layers
        self.parameter_list = []

        # layers_ = self.layers
        # self.encoder = tf.keras.Sequential()
        # self.encoder.add(tf.keras.Input(shape=(layers_[0], )))
        # for i, n_units in enumerate(layers_[1:]):
        #     self.encoder.add(layers.Dense(n_units, activation='sigmoid'))

        # layers_ = self.layers[::-1]
        # self.decoder = tf.keras.Sequential()
        # self.decoder.add(tf.keras.Input(shape=(layers_[0], )))
        # for i, n_units in enumerate(layers_[1:]):
        #     self.decoder.add(layers.Dense(n_units, activation='sigmoid'))

        # self.autoencoder = tf.keras.Sequential([self.encoder, self.decoder])
        initializer = tf.initializers.glorot_uniform()
        zero_init = tf.initializers.zeros()
        # Hidden layers:
        layers = self.layers[1:]
        last_layer_size = self.layers[0]
        for i, layer in enumerate(layers):
            # self.encoding_W.append(tf.Variable(
            #     tf.random.normal(shape=[last_layer_size, layer]), name='e-W%d' % i, trainable=True, dtype=tf.float32))
            # self.encoding_b.append(tf.Variable(
            #     tf.random.normal(shape=[layer]), name='e-b%d' % i, trainable=True))
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
            # self.decoding_W.append(tf.Variable(
            #     tf.random.normal(shape=[last_layer_size, layer]), name='d-W%d' % i, trainable=True, dtype=tf.float32))
            # self.decoding_b.append(tf.Variable(
            #     tf.random.normal(shape=[layer]), name='d-b%d' % i, trainable=True))
            self.decoding_W.append(tf.Variable(
                initializer(shape=[last_layer_size, layer]), name='d-W%d' % i, trainable=True, dtype=tf.float32))
            self.decoding_b.append(tf.Variable(
                zero_init(shape=[layer]), name='d-b%d' % i, trainable=True))
            self.decoding_layers.append(lambda X, j=i: tf.nn.sigmoid(X @ self.decoding_W[j] + self.decoding_b[j]))
            self.parameter_list.append(self.decoding_W[i])
            self.parameter_list.append(self.decoding_b[i])
            last_layer_size = layer

        return

    # Eqn. 4
    def loss(self, omega, S, X_pred):
        return tf.reduce_sum(
                    S * tf.square(tf.norm(
                        omega - tf.expand_dims(X_pred, axis=1), axis=2)))

    def gradients(self, X, omega, S):
        with tf.GradientTape() as tape:
            X_pred = self.predict(X, training=True)
            loss_value = self.loss(omega, S, X_pred)
            # grads = tape.gradient(loss_value, self.parameter_list)
            grads = tape.gradient(loss_value, self.parameter_list)
            return loss_value, grads

    def preprocess(self, data):
        data = data.map(lambda X_batch, y_batch: (X_batch, y_batch, None))
        return data

    def fit(self, data, epochs=10000): 
        # epoch = expected number of iterations until convergence
        with self.file_writer.as_default():
            # 1. Compute the reconstruction weights Si from {x_i} and
            #    determine the reconstruction set i, e.g. by k-nearest neighbor
            data = self.preprocess(data)

            for epoch in range(epochs):
                # 2. Minimize E in Eqn.4 using the stochastic gradient 
                #    descent and update theta for t steps.
                loss_value = 0.
                for X_batch, y_batch, encoded_batch in data:
                    omega_batch = self.compute_reconstruction_set(encoded_batch, X_batch, y_batch)
                    S_batch = self.compute_reconstruction_weights(encoded_batch, X_batch, y_batch)
                    
                    with tf.GradientTape() as tape:
                        X_pred = self.predict(X_batch, training=True)
                        # X_pred = self.autoencoder(X_batch, training=True)
                        loss_value = self.loss(omega_batch, S_batch, X_pred)
                    grads = tape.gradient(loss_value, self.parameter_list)
                    self.optimizer.apply_gradients(zip(grads, self.parameter_list)) 
                    # grads = tape.gradient(loss_value, self.autoencoder.trainable_variables)
                    # self.optimizer.apply_gradients(zip(grads, self.autoencoder.trainable_variables)) 
                
                # 3. Compute the hidden representation y_i, and update S_i and omega_i from y_i
                if self.recalculate_reconstruction_sets:
                    # X = X.map(lambda X_batch, y_batch, encoded: (X_batch, y_batch, self.encode(X_batch)))
                    data = self.recompute(data)
                    
                if epoch % 1 == 0:
                    tf.summary.scalar('loss', loss_value, step=epoch)
                    tf.print("Epoch: ", epoch, "loss_value=", loss_value)

    def recalculate_reconstruction(self, X_batch, y_batch, omega_batch, s_batch):
        # omega and s are discarded and recalculated
        encoded = self.encode(X_batch)
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
        for X_batch, y_batch, encoded_batch in X:
            for i, x in enumerate(X_batch):
                class_label = self.get_class_label(y_batch[i])
                classes[class_label].append(x)
        
        classes = [tf.RaggedTensor.from_tensor(c) for c in classes]
        return classes

    def evaluate(self):
        pass

class DeepPCA(DeepGAE):
    def __init__(self, file_writer):
        super(DeepPCA, self).__init__(file_writer)
        self.recalculate_reconstruction_sets = False
        self.model_name = "PCA"

    @tf.function
    def compute_reconstruction_set(self, encoded_batch, X_batch, y_batch):
        return tf.map_fn(fn=lambda x: tf.convert_to_tensor([x]), elems=X_batch, 
            fn_output_signature=tf.TensorSpec(shape=(1, X_batch.shape[1])))

    @tf.function
    def compute_reconstruction_weights(self, encoded_batch, X_batch, y_batch):
        return tf.map_fn(fn=lambda x: tf.convert_to_tensor([1.0], dtype=tf.float32), elems=X_batch, 
            fn_output_signature=tf.TensorSpec(shape=(1,)))

class DeepLDA(DeepGAE):
    def __init__(self, file_writer):
        super(DeepLDA, self).__init__(file_writer)
        self.recalculate_reconstruction_sets = False
        self.omega_classes = None
        self.model_name = "LDA"

    def preprocess(self, X, y):
        X = X.map(lambda X_batch, y_batch: (X_batch, y_batch, None))
        self.omega_classes = self.get_class_division(X, y)
        return X, y

    def loss(self, omega, S, X_pred):
        s = 0
        for i, omega_i in enumerate(omega):
            s += tf.reduce_sum(
                    S[i] * tf.square(tf.norm(
                        omega_i.to_tensor() - X_pred[i], axis=1)))
        return s
    
    def compute_reconstruction_set(self, encoded_batch, X_batch, y_batch):
        mappingFunc = lambda i: self.omega_classes[
                    self.get_class_label(y_batch[i.numpy()])]
        return tf.map_fn(
            fn=mappingFunc,
            elems=tf.range(X_batch.shape[0]),
            dtype=X_batch.dtype,
            fn_output_signature=tf.RaggedTensorSpec(shape=[None, X_batch.shape[1]]))

    def compute_reconstruction_weights(self, encoded_batch, X_batch, y_batch):
        mappingFunc = lambda i: tf.convert_to_tensor(
                    # Compute weight which is inversely proportional to class size
                    [1. / self.omega_classes[
                        self.get_class_label(y_batch[i])].shape[0]], 
                    dtype=tf.float32
                )

        return tf.map_fn(
            fn=mappingFunc, 
            elems=tf.range(X_batch.shape[0]), 
            fn_output_signature=tf.RaggedTensorSpec(shape=(1,)))

class DeepMFA(DeepGAE):
    def __init__(self, layers, file_writer, k=18):
        super(DeepMFA, self).__init__(layers, file_writer)
        self.recalculate_reconstruction_sets = True
        self.omega_classes = None
        self.k = k
        self.model_name = "MFA"

    def get_encoded_class_division(self, X):
        classes = [[] for i in range(self.n_classes)]
        for X_batch, y_batch, encoded_batch in X:
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


    def recompute(self, data):
        data = data.map(lambda X_batch, y_batch, encoded_batch: (X_batch, y_batch, self.encode(X_batch)))

        self.omega_encoded_classes = np.array(self.get_encoded_class_division(data))
        self.omega_encoded_classes_complement = []
        for Ci, X_class in enumerate(self.omega_encoded_classes):
            class_inx=list(range(self.n_classes))
            class_inx.pop(Ci)
            self.omega_encoded_classes_complement.append(
                tf.concat(self.omega_encoded_classes[class_inx].tolist(), axis=0).to_tensor())

        return data

    
    def knn(self, xj, data):
        distances = tf.norm(data - xj, axis=1)
        _, top_k_indices = tf.nn.top_k(tf.negative(distances), k=self.k)
        # return tf.gather(data, top_k_indices)
        return top_k_indices
        
    
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
            omega_k1_inx = self.knn(X_batch_[i], omega_classes_[Ci].to_tensor())
            omega_k2_inx = self.knn(X_batch_[i], omega_classes_complement_[Ci])
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

def dataset(batch_size, num_sets = 30, validation_ratio = 0.2):
    MNIST_size = 60000

    def _dataset(X, y):
        y = tf.one_hot(y, depth=10)
        X = tf.data.Dataset.from_tensor_slices((X, y)) \
            .map(preprocess) \
            .shuffle(10000)
        return X

    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

    train, test = _dataset(X_train, y_train), _dataset(X_test, y_test)

    set_size = math.floor(MNIST_size/num_sets)
    train_size = math.floor(set_size * (1 - validation_ratio))
    valid_size = math.floor(set_size * validation_ratio)

    train_valid = []
    for i in range(30):
        _train = train.skip(i*set_size).take(train_size).batch(batch_size)
        _valid = train.skip(i*set_size + train_size).take(valid_size).batch(batch_size)
        train_valid.append((_train,_valid))

    test = test.batch(batch_size)
    return train_valid, test

def get_class_label(y): #possibly shouldn't be member of class
        return tf.argmax(y, axis=0).numpy()

def knn(k, xj, data):
    distances = tf.norm(data - xj, axis=1)
    _, top_k_indices = tf.nn.top_k(tf.negative(distances), k=k)
    return top_k_indices 

def factory(model_name):
    logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    file_writer = tf.summary.create_file_writer(logdir)
    if model_name == "PCA":
        return DeepPCA(file_writer)
    elif model_name == "LDA":
        return DeepLDA(file_writer)
    elif model_name == "MFA":
        return DeepMFA(file_writer)



@tf.autograph.experimental.do_not_convert
def main(argc, argv):
    image_shape = (28, 28)
    load_existing_model = False
    input_size = image_shape[0] * image_shape[1]
    n_classes = 10
    batch_size = 64

    epochs = [500]
    samples = [2500]
    layers_configs = [[input_size, 200, 100, 2],[input_size, 400, 200, 2],[input_size, 800, 400, 2],[input_size, 800, 400, 200, 2]]
    learning_rates = [0.001]
    momentums = [0.9]
    model_names = ["PCA"]

    train_valid_set, test_set = dataset(batch_size)

    with tf.device('CPU:0'):
        i = 0
        for n_epochs in epochs:
            for n_samples in samples:
                for layers in layers_configs:
                    for learning_rate in learning_rates:
                        for momentum in momentums:
                            for model_name in model_names:
                                layer_string = '-'.join(str(x) for x in layers)
                                print("########################################################")
                                print("Validating config: %s model - %d epochs - %d samples - %s layers - %f learning rate - %f momentum - %d batch size" % (model_name, n_epochs, n_samples, layer_string, learning_rate, momentum, batch_size))
                                train_set, valid_set = train_valid_set[i]

                                save_path = "_".join((".models/",model_name,str(n_epochs),str(batch_size),str(n_samples),str(layer_string),str(learning_rate),str(momentum)))

                                if os.path.exists(save_path) and load_existing_model:
                                    print("Loading existing model...")
                                    model = load_model(save_path)
                                    
                                else:
                                    print("Training new model...")
                                    
                                    model = factory(model_name)
                                    # Note! We use keras optimizer.
                                    # TODO: try with momentum..
                                    # learning_rate=0.01, momentum=0.9
                                    model.compile(optimizers.SGD(learning_rate=learning_rate, momentum=momentum), layers, n_classes)
                                    model.fit(train_set, epochs=n_epochs)
                                    print("Saving model...")
                                    # return 0
                                    save_model(model, save_path)
                                    os.mkdir(save_path+"/img")


                                #ERROR RATE and IMPURITY
                                er_k = 18 #choose k+1 as the instance itself will also be there.
                                valid_set_enc_arr = []
                                valid_set_label_arr = []
                                for y,e in valid_set.map(lambda x,y: (y,model.encode(x))).unbatch():
                                    valid_set_enc_arr.append(e)
                                    valid_set_label_arr.append(y)
                            
                                errors = 0
                                impurities = 0
                                for i, x_enc in enumerate(valid_set_enc_arr):
                                    top_k_indices = knn(er_k, x_enc, valid_set_enc_arr)
                                    label = get_class_label(valid_set_label_arr[i])
                                    votes_against = 0
                                    for index in top_k_indices:
                                        if label != get_class_label(valid_set_label_arr[index]):
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

                                # 2D GRAPH
                                fig_, ax = plt.subplots()
                                class_points_x = [[] for i in range(10)]
                                class_points_y = [[] for i in range(10)]
                                for i, (x,y) in enumerate(valid_set.map(lambda x,y: (model.encode(x), y)).unbatch().map(lambda x,y: (postprocess(x, (1,2)), y))):
                                    coord = x.numpy().ravel()
                                    label = get_class_label(y)
                                    class_points_x[label].append(coord[0])
                                    class_points_y[label].append(coord[1])

                                for label in range(10):
                                    ax.scatter(class_points_x[label], class_points_y[label], label="%d" % label)

                                plt.legend()
                                plt.savefig(save_path+"/img/2d_encode.png")

                                # Reconstruct test samples using model:
                                Xr = valid_set.unbatch() \
                                    .map(lambda X, y: (postprocess(X, image_shape), X)) \
                                    .batch(batch_size) \
                                    .map(lambda Xp, X: (Xp, model.predict(X), model.encode(X))) \
                                    .unbatch() \
                                    .map(lambda Xp, X, Xen: (Xp, postprocess(X), postprocess(Xen, (1, 2)))) \
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

                                plt.savefig(save_path+"/img/predict.png")

    return 0


if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))