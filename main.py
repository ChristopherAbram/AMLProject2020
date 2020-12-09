import numpy as np
import sys, os, math

import tensorflow as tf
from tensorflow.keras import datasets, optimizers
import math
import itertools as it

# tf.python.framework.ops.disable_eager_execution()
tf.config.run_functions_eagerly(False)

from autoencoder.nn import factory, load_model
from autoencoder.validation import *

print('Intra op threads: {}'.format(tf.config.threading.get_intra_op_parallelism_threads()))
print('Inter op threads: {}'.format(tf.config.threading.get_inter_op_parallelism_threads()))

if(tf.executing_eagerly()):
    print('Eager execution is enabled (running operations immediately)\n')
else:
    print('Eager execution is off\n')

print(('\nYour devices that are available:\n{0}').format(
    [device.name for device in tf.config.experimental.list_physical_devices()]
))

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


def main(argc, argv):
    # Constant huperparametrs:
    image_shape = (28, 28)
    load_existing_model = False
    input_size = image_shape[0] * image_shape[1]
    n_classes = 10
    batch_size = 64
    n_samples = 10000
    pretrain_epoch_ratio = 0.1
    validation_ratio = 0.2
    # How often run validation in model.fit(), e.g. 2 means run evaluate every 2 epochs:
    validation_freq = 1
    # To disable validation in model.fit(), pass None as validation_data
    # In our case set run_validation_per_epoch to False if you don't need to evaluate during training
    run_validation_per_epoch = True # used only in main loop

    # Lists of hyperparametrs:
    epochs = [50]
    layers_configs = [[input_size, 500, 200, 30]]
    learning_rates = [0.003]
    momentums = [0.3]
    model_names = ['LDA']

    # Create all compination of hyperparametr lists:
    configs = it.product(model_names, epochs, layers_configs, learning_rates, momentums)

    # with tf.device('CPU:0'):
    train_valid_set, test_set = dataset(MNIST_size // n_samples, validation_ratio)
    for i, (model_name, n_epochs, layers, learning_rate, momentum) in enumerate(configs):
        layer_string = '-'.join(str(x) for x in layers)
        print("########################################################")
        print("Validating config: %s model - %d epochs - %d samples - %s layers - %f learning rate - %f momentum - %d batch size" % 
            (model_name, n_epochs, n_samples, layer_string, learning_rate, momentum, batch_size))
        (X_train, y_train), (X_valid, y_valid) = train_valid_set[i]

        save_path = ".models/" + "_".join(
            (model_name, str(n_epochs), str(batch_size), str(n_samples), 
                str(layer_string), str(learning_rate), str(momentum)))

        if os.path.exists(save_path) and load_existing_model:
            print("Loading existing model...")
            model = load_model(save_path)
            
        else:
            print("Training new model...")
            
            model = factory(model_name)
            # Note! We use keras optimizer.
            # momentum=momentum
            model.compile(optimizers.RMSprop(learning_rate=learning_rate, momentum=momentum), layers, n_classes)

            validation_data = (X_valid, y_valid) if run_validation_per_epoch else None
            history = model.fit(X_train, y_train, 
                epochs=n_epochs, 
                pretrain_epochs=(pretrain_epoch_ratio * n_epochs),
                validation_freq=validation_freq,
                validation_data=validation_data)
            
            print("Saving model...")        
            model.save(save_path)

            if not os.path.exists(save_path + "/img"):
                os.mkdir(save_path + "/img")

            # Plot loss function and other metrics:
            if not run_validation_per_epoch:
                plot_history(history, 'loss', filepath=os.path.join(save_path, 'img', 'loss.png'))
            else:
                plot_history_2combined(history, 'loss', 'val_loss', filepath=os.path.join(save_path, 'img', 'loss.png'))
                plot_history(history, 'error_rate', filepath=os.path.join(save_path, 'img', 'error_rate.png'))
                plot_history(history, 'impurity', filepath=os.path.join(save_path, 'img', 'impurity.png'))

        error_rate, impurity = model.evaluate(X_valid, y_valid)
        print("ERROR RATE: %f%%" % error_rate)
        print("AVG. IMPURITY: %f" % impurity)
        info_file = open(save_path+"/info.txt","w+")
        info_file.write("ERROR RATE: %f%%" % error_rate)
        info_file.write("AVG. IMPURITY: %f" % impurity)
        info_file.close()


        X_valid_encoded = model.encode(X_valid)
        X_valid_predicted = model.predict(X_valid)
        width, height = encoded_display_shape(layers[len(layers) - 1])

        # 2D GRAPH
        if width == 2:
            scatter_plot_2d(save_path, X_valid_encoded, X_valid, y_valid, n_classes)

        # Visualize reconstructed images:
        plot_col, plot_row = 3, 9
        X_valid = X_valid[:plot_col * plot_row]
        X_valid_encoded = X_valid_encoded[:plot_col * plot_row]
        X_valid_predicted = X_valid_predicted[:plot_col * plot_row]

        plot_table(save_path, X_valid, X_valid_predicted, X_valid_encoded, 
            image_shape, (width, height), (plot_row, plot_col))

    return 0


if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))