from inspect import stack
import numpy as np
import tensorflow as tf
import os
from datetime import datetime
from autoencoder.validation import error_rate_impurity, tf_error_rate_impurity

class DeepGAE(tf.Module):
    """
    Deep Generalized Autoencoder, abstract class with common implementation for all methods.
    """
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
        """
        Encodes the input X by mapping the input through 
        the hidden encoding layers to a hidden representation Y.
        """
        Y = X
        for layer in self.encoding_layers:
            Y = layer(Y)
        return Y
    
    @tf.function
    def decode(self, Y, training=False):
        """
        Reconstructs the input X' from the hidden 
        representation Y through the symmetric hidden decoding layers.
        """
        X = Y
        for layer in self.decoding_layers:
            X = layer(X)
        return X

    @tf.function
    def __call__(self, X):
        return self.predict(X)

    def compile(self, optimizer, layers, n_classes):
        """
        Builds and initilizes all layers for both encoder and decoder, 
        sets the optimizer and builds a list of parameters references for optimizer.
        """
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

    @tf.function
    def loss(self, omega, S, X_pred):
        """
        Total linear reconstruction error by Equation 4.
        """
        return tf.reduce_sum(
                    S * tf.square(tf.norm(
                        omega - tf.expand_dims(X_pred, axis=1), axis=2)))

    def preprocess(self, X, y, batch_size=64):
        """
        Converts numpy array into tf.data.Dataset, shuffles and batches all the input.
        """
        y_ = tf.one_hot(y, depth=self.n_classes)
        return tf.data.Dataset.from_tensor_slices((X, y_)) \
            .shuffle(10000) \
            .batch(batch_size) \
            .map(lambda X_batch, y_batch: (X_batch, y_batch, None))

    def preprocess_validation_data(self, X, y, batch_size=64):    
        y_ = tf.one_hot(y, depth=self.n_classes)
        return tf.data.Dataset.from_tensor_slices((X, y_)) \
            .shuffle(10000) \
            .batch(batch_size)

    def recompute(self, data):
        """
        Default behavior of dGAE is to not recompute reconstruction sets for each epoch.
        """
        return data

    def fit(self, X, y, epochs=10000, 
            pretrain_epochs=1, 
            batch_size=64, 
            initial_epoch=1, 
            steps_per_epoch=None, 
            validation_data=None, 
            validation_freq=1): 
        """
        Run training for given data and hyperparameters.
        Returns history of training.
        """
        # epochs = expected number of iterations until convergence
        with self.file_writer.as_default():
            # 1. Compute the reconstruction weights Si from {x_i} and
            #    determine the reconstruction set i, e.g. by k-nearest neighbor
            data_ = self.preprocess(X, y, batch_size)
            validation_data_ = self.preprocess_validation_data(validation_data[0], validation_data[1], batch_size) \
                if validation_data != None else None

            history = { 'loss': [], 'val_loss': [], 'error_rate': [], 'impurity': [] }

            for epoch in range(initial_epoch, epochs + 1):
                # 2. Minimize E in Eqn.4 using the stochastic gradient 
                #    descent and update theta for t steps.
                total_training_loss = 0.
                avg_training_loss = 0.
                total_validation_loss = 0.
                avg_validation_loss = 0.

                # Display a progress bar
                num_steps_ = X.shape[0] // batch_size
                printProgressBar(0, num_steps_, 
                    prefix=f'Epochs {epoch}/{epochs}', suffix='', length=50)

                # Training loop
                for train_step, (X_batch, y_batch, encoded_batch) in enumerate(data_):
                    # Compute initial reconstruction weights and reconstruction set:
                    omega_batch, S_batch = self.compute_reconstruction(
                        encoded_batch if epoch > pretrain_epochs else None, X_batch, y_batch)
                    
                    # Minimize E in equation 4:
                    with tf.GradientTape() as tape:
                        X_pred = self.predict(X_batch, training=True)
                        loss_value = self.loss(omega_batch, S_batch, X_pred)
                    grads = tape.gradient(loss_value, self.parameter_list)
                    self.optimizer.apply_gradients(zip(grads, self.parameter_list))

                    total_training_loss += loss_value
                    avg_training_loss = total_training_loss / (train_step + 1)

                    printProgressBar(train_step + 1, num_steps_, 
                        prefix=f'Epochs {epoch}/{epochs}', 
                        suffix='loss= {:.4f}'.format(avg_training_loss.numpy()), length=50, end=False)

                # Validation loop
                if validation_data_ is not None:
                    for val_step, (X_batch, y_batch) in enumerate(validation_data_):
                        encoded_batch = self.encode(X_batch)
                        omega_batch, S_batch = self.compute_reconstruction(
                            encoded_batch, X_batch, y_batch)
                        
                        X_pred = self.decode(encoded_batch)
                        loss_value = self.loss(omega_batch, S_batch, X_pred)
                        total_validation_loss += loss_value
                        avg_validation_loss = total_validation_loss / (val_step + 1)

                    # Compute metrics and insert to the log:
                    if validation_freq > 0 and (epoch % validation_freq == 0 or epoch == 1):
                        error_rate, impurity = self.evaluate(validation_data[0], validation_data[1])
                        history['error_rate'].append([epoch, error_rate])
                        history['impurity'].append([epoch, impurity])
                        tf.summary.scalar('error_rate', error_rate, step=epoch)
                        tf.summary.scalar('impurity', impurity, step=epoch)

                        printProgressBar(train_step + 1, num_steps_, 
                            prefix=f'Epochs {epoch}/{epochs}', 
                            suffix='loss= {:.4f}, val_loss= {:.4f}, error_rate= {:.2f}, impurity= {:.3f}'.format(
                                avg_training_loss.numpy(), avg_validation_loss.numpy(), 
                                error_rate, impurity), length=50, end=False)
                        print()

                    history['val_loss'].append([epoch, avg_validation_loss.numpy()])
                    tf.summary.scalar('val_loss', avg_validation_loss, step=epoch)

                else: # new line for progress bar
                    print()

                history['loss'].append([epoch, avg_training_loss.numpy()])
                tf.summary.scalar('loss', avg_training_loss, step=epoch)
                
                # 3. Compute the hidden representation y_i, and update S_i and omega_i from y_i
                if self.recalculate_reconstruction_sets:
                    data_ = self.recompute(data_)

        return history

    @tf.function
    def get_class_label(self, y):
        """
        Convert one-hot categorization vector to class label (number)
        """
        return tf.argmax(y, axis=0)

    def get_class_division(self, X, y, equal_sizes=True):
        """
        Divide given instances into lists indexed by ther corresponding class label.
        If equal_sizes is true, all classes will have the same number of instances, 
        truncated to match the smallest class, otherwise ragged tensors are used.
        """
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
        """
        Stores model on the disk.
        """
        tf.saved_model.save(self, filepath)

    def evaluate(self, X, y):
        """
        Evaluate the model with selected metrics
        """
        error_rate, impurity = tf_error_rate_impurity(self.encode(X), X, y, 18)
        return error_rate.numpy(), impurity.numpy()
    
def load_model(filepath):
    """
    Loads existing model from disk.
    """
    return tf.saved_model.load(filepath)

class DeepPCA(DeepGAE):
    """
    Implementation of dGAE-PCA
    """
    def __init__(self, file_writer):
        super(DeepPCA, self).__init__(file_writer)
        self.recalculate_reconstruction_sets = False

    @tf.function
    def compute_reconstruction(self, encoded_batch, X_batch, y_batch):
        """
        Calcualte reconstruction set and weights. In case of dGAE-PCA, this is the element itself with weight 1.
        """
        omega = tf.map_fn(fn=lambda x: tf.convert_to_tensor([x]), elems=X_batch, 
            fn_output_signature=tf.TensorSpec(shape=(1, X_batch.shape[1])))
        S = tf.map_fn(fn=lambda x: tf.convert_to_tensor([1.0], dtype=tf.float32), elems=X_batch, 
            fn_output_signature=tf.TensorSpec(shape=(1,)))
        return omega, S

class DeepBalancedLDA(DeepGAE):
    """
    A variation of dGAE-LDA that is not ragged. 
    I.e. all classes used for reconstruction set is truncated to match the size of the smallest class.
    Used for experimental purposes.
    """
    def __init__(self, file_writer):
        super(DeepBalancedLDA, self).__init__(file_writer)
        self.recalculate_reconstruction_sets = False
        self.omega_classes = None

    def preprocess(self, X, y, batch_size=64):
        """
        dGAE-LDA calculates reconstruction set and weights before the loop for efficiency.
        """
        self.omega_classes, self.omega_counts = self.get_class_division(X, y)
        return super(DeepBalancedLDA, self).preprocess(X, y, batch_size)
    
    @tf.function
    def compute_reconstruction(self, encoded_batch, X_batch, y_batch):
        """
        Reconstruction set is calculated by including all elements within the same class,
        maximally the same number of elements as the smallest class.
        The reconstruction weight is 1/n where n is the number of elements of the smallest class.
        """
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
    """
    A normal variation of dGAE-LDA with ragged tensors. 
    """
    def __init__(self, file_writer):
        super(DeepLDA, self).__init__(file_writer)
        self.recalculate_reconstruction_sets = False
        self.omega_classes = None

    def preprocess(self, X, y, batch_size):
        """
        dGAE-LDA calculates reconstruction set and weights before the loop for efficiency.
        """
        data = super(DeepLDA, self).preprocess(X, y, batch_size)
        self.omega_classes = self.get_class_division(data)
        self.omega_counts = tf.cast(tf.convert_to_tensor([cl.shape[0] for cl in self.omega_classes]), tf.float32)
        self.omega_classes = stack_ragged(self.omega_classes)
        return data

    def get_class_division(self, data):
        """
        Divide given instances into lists indexed by ther corresponding class label.
        """
        classes = [[] for i in range(self.n_classes)]
        for X_batch, y_batch, encoded_batch in data:
            for i, x in enumerate(X_batch):
                class_label = self.get_class_label(y_batch[i])
                classes[class_label].append(x)
        classes = [tf.convert_to_tensor(c) for c in classes]
        return classes

    @tf.function
    def loss(self, omega, S, X_pred):
        """
        The loss function for dGAE-LDA differs to the others, 
        as ragged tensors are used to support classes of different sizes.
        """
        return tf.reduce_sum(
            S * tf.square(tf.norm(
                (omega - tf.expand_dims(X_pred, axis=1)).to_tensor(), axis=2)))
    
    @tf.function
    def compute_reconstruction(self, encoded_batch, X_batch, y_batch):
        """
        Reconstruction set is calculated by including all elements within the same class.
        The reconstruction weight is 1/n where n is the number of elements of the class.
        """
        omega = tf.gather(self.omega_classes, tf.argmax(y_batch, axis=1))
        S = tf.map_fn(
            fn=lambda y: tf.convert_to_tensor(
                [1. / tf.gather(self.omega_counts, self.get_class_label(y))]),
            elems=y_batch, 
            fn_output_signature=tf.TensorSpec(shape=(1,), dtype=tf.float32))
        return omega, S

class DeepEagerLDA(DeepGAE):
    """
    Version of LDA which runs only in eager mode, used for debugging and experimentation.
    """
    def __init__(self, file_writer):
        super(DeepEagerLDA, self).__init__(file_writer)
        self.recalculate_reconstruction_sets = False
        self.omega_classes = None

    def preprocess(self, X, y, batch_size):
        data = super(DeepEagerLDA, self).preprocess(X, y, batch_size)
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
        """
        The loss function is overriden due to uncompabilities 
        of reconstruction set with the default impl. of the loss.
        """
        s = 0
        for i, omega_i in enumerate(omega):
            s += tf.reduce_sum(
                    S[i] * tf.square(tf.norm(
                        omega_i.to_tensor() - X_pred[i], axis=1)))
        return s
    
    @tf.function
    def compute_reconstruction(self, encoded_batch, X_batch, y_batch):
        """
        Computes both reconstruction set omega and associated weights S.
        The omega set consists of class instances, wheres the weight are the inverses of the class counts.
        """
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
    """
    Implementation of the dGAE-LE.
    """
    def __init__(self, file_writer, k=10):
        super(DeepLE, self).__init__(file_writer)
        self.recalculate_reconstruction_sets = True
        self.k = k
        self.t = 1.2
        self.data = None
        self.data_encoded = None

    def preprocess(self, X, y, batch_size=64):
        self.data = tf.convert_to_tensor(X)
        self.data_encoded = self.encode(self.data)
        return super(DeepLE, self).preprocess(X, y, batch_size)

    def recompute(self, data):
        data = data.map(lambda X_batch, y_batch, encoded_batch: (X_batch, y_batch, self.encode(X_batch)))
        self.data_encoded = self.encode(self.data)
        return data

    @tf.function
    def compute_reconstruction(self, encoded_batch, X_batch, y_batch):
        """
        Recomputes the reconstruction weights and set on the hidden representation with k-nearest neighbor.
        """
        X_batch_ = X_batch if encoded_batch is None else encoded_batch
        data_ = self.data if encoded_batch is None else self.data_encoded
        omega = tf.map_fn(
            fn=lambda x: tf.gather(self.data, knn(x, data_, k=self.k)),
            elems=X_batch_, 
            fn_output_signature=tf.TensorSpec(shape=(self.k, X_batch.shape[1])))
        S = tf.exp(tf.negative(
                tf.square(tf.norm(
                    tf.expand_dims(X_batch, axis=1) - omega, axis=2)) / self.t))
        return omega, S

def stack_ragged(classes):
    """
    Stacks list of ragged tensor into one ragged tensor.
    """
    values = tf.concat(classes, axis=0)
    lens = tf.stack([tf.shape(cl, out_type=tf.int64)[0] for cl in classes])
    return tf.RaggedTensor.from_row_lengths(values, lens)

class DeepMFA(DeepGAE):
    """
    Implementation of dGAE-MFA.
    """
    def __init__(self, file_writer, k1=15, k2=5):
        super(DeepMFA, self).__init__(file_writer)
        self.recalculate_reconstruction_sets = True
        self.omega_classes = None
        self.k1 = k1
        self.k2 = k2

    def get_encoded_class_division(self, data):
        """
        Divide given _encoded_ instances into lists indexed by ther corresponding class label.
        """
        classes = [[] for i in range(self.n_classes)]
        for X_batch, y_batch, encoded_batch in data:
            for i, x in enumerate(encoded_batch):
                class_label = self.get_class_label(y_batch[i])
                classes[class_label].append(x)
        classes = [tf.convert_to_tensor(c) for c in classes]
        return classes

    def get_class_division(self, data):
        """
        Divide given instances into lists indexed by ther corresponding class label.
        """
        classes = [[] for i in range(self.n_classes)]
        for X_batch, y_batch, encoded_batch in data:
            for i, x in enumerate(X_batch):
                class_label = self.get_class_label(y_batch[i])
                classes[class_label].append(x)
        classes = [tf.convert_to_tensor(c) for c in classes]
        return classes

    def preprocess(self, X, y, batch_size):
        """
        dGAE-MFA calculates class division and complements of class divisions before training loop for efficiency.
        """
        data = super(DeepMFA, self).preprocess(X, y, batch_size)
        # 1. group points into classes
        self.omega_classes = np.array(self.get_class_division(data))
        self.omega_classes_complement = []
        for Ci, X_class in enumerate(self.omega_classes):
            class_inx=list(range(self.n_classes))
            class_inx.pop(Ci)
            self.omega_classes_complement.append(
                tf.concat(self.omega_classes[class_inx].tolist(), axis=0))
        self.omega_classes = stack_ragged(self.omega_classes.tolist())
        self.omega_classes_complement = stack_ragged(self.omega_classes_complement)
        data_ = self.recompute(data)
        return data

    def recompute(self, data):
        data = data.map(lambda X_batch, y_batch, encoded_batch: (X_batch, y_batch, self.encode(X_batch)))
        self.omega_encoded_classes = np.array(self.get_encoded_class_division(data))
        self.omega_encoded_classes_complement = []
        for Ci, X_class in enumerate(self.omega_encoded_classes):
            class_inx=list(range(self.n_classes))
            class_inx.pop(Ci)
            self.omega_encoded_classes_complement.append(
                tf.concat(self.omega_encoded_classes[class_inx].tolist(), axis=0))
        self.omega_encoded_classes = stack_ragged(self.omega_encoded_classes.tolist())
        self.omega_encoded_classes_complement = stack_ragged(self.omega_encoded_classes_complement)
        return data

    @tf.function
    def mappingFunc(self, x, omega_classes, omega_classes_complement):
        """
        A function for mapping an element to its reconstruction set.
        """
        x_ = x[:-10]
        label = self.get_class_label(x[-10:])
        omega_k1_inx = knn(x_, omega_classes[label], k=self.k1)
        omega_k2_inx = knn(x_, omega_classes_complement[label], k=self.k2)
        return tf.concat([
            tf.gather(self.omega_classes[label], omega_k1_inx), 
            tf.gather(self.omega_classes_complement[label], omega_k2_inx)], axis=0)
    
    @tf.function
    def compute_reconstruction(self, encoded_batch, X_batch, y_batch):
        """
        Recomputes the reconstruction weights and set on the hidden representation with k1- and k2-nearest neighbor.
        """
        X_batch_ = X_batch if encoded_batch is None else encoded_batch
        omega_classes_ = self.omega_classes if encoded_batch is None else self.omega_encoded_classes
        omega_classes_complement_ = self.omega_classes_complement \
            if encoded_batch is None else self.omega_encoded_classes_complement
        omega = tf.map_fn(
            fn=lambda x: self.mappingFunc(x, omega_classes_, omega_classes_complement_),
            elems=tf.concat(
                [X_batch_, tf.cast(y_batch, X_batch_.dtype)], axis=1),
            fn_output_signature=tf.TensorSpec(shape=(self.k1 + self.k2, X_batch.shape[1])))
        S = tf.map_fn(
            fn=lambda y: \
                tf.convert_to_tensor(
                    ([1.0] * self.k1) + ([-1.0] * self.k2), dtype=tf.float32),
            elems=y_batch, 
            fn_output_signature=tf.TensorSpec(shape=(self.k1 + self.k2, )))
        return omega, S

def factory(model_name):
    """
    A factory for initialising various version of dGAE.
    """
    logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    file_writer = tf.summary.create_file_writer(logdir)
    if model_name == "PCA":
        return DeepPCA(file_writer)
    elif model_name == "BalancedLDA":
        return DeepBalancedLDA(file_writer)
    elif model_name == "LDA":
        return DeepLDA(file_writer)
    elif model_name == "EagerLDA":
        return DeepEagerLDA(file_writer)
    elif model_name == "MFA":
        return DeepMFA(file_writer)
    elif model_name == "LE":
        return DeepLE(file_writer)

@tf.function
def knn(xj, data, k=18):
    """
    k-nearest neighbor for TensorFlow tensors, where xj is a tensor and data is a tensor of tensors of the same size.
    """
    distances = tf.norm(data - xj, axis=1)
    _, top_k_indices = tf.nn.top_k(tf.negative(distances), k=k)
    return top_k_indices

# Method for printing a pretty pregressbar when training the network
# Kindly borrowed from: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r", end=True):
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
    
    if iteration == total and end: 
        print()