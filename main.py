import numpy as np
import sys, os, math

import tensorflow as tf
from tensorflow.keras import datasets, optimizers
import math

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
    image_shape = (28, 28)
    load_existing_model = False
    input_size = image_shape[0] * image_shape[1]
    n_classes = 10
    batch_size = 64
    n_samples = 2000

    epochs = [10]
    layers_configs = [[input_size, 200, 100, 2]]
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
                                model = load_model(save_path)
                                
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

                            X_valid_encoded = model.encode(X_valid)
                            X_valid_predicted = model.predict(X_valid)

                            error_rate, impurity = error_rate_impurity(X_valid_encoded, X_valid, y_valid, k=18)
                            print("ERROR RATE: %f%%" % error_rate)
                            print("AVG. IMPURITY: %f" % impurity)
                            info_file = open(save_path+"/info.txt","w+")
                            info_file.write("ERROR RATE: %f%%" % error_rate)
                            info_file.write("AVG. IMPURITY: %f" % impurity)
                            info_file.close()

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