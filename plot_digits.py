import sys, os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, optimizers

tf.config.run_functions_eagerly(True)

import autoencoder.nn as ann
# import autoencoder.validation as aval

def select_random_digits_from_each_class(X, y):
    classes = []
    digits = []
    np_digits = np.zeros((64, X[0].shape[0]), dtype=np.float32)
    for Ci in range(10):
        classes.append(X[np.where(y==Ci)])
        np.random.shuffle(classes[Ci])
        digits.append(classes[Ci][0])
    np_digits[:10,:] = np.array(digits)
    return np_digits

def main(argc, argv):
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

    # Preprocess:
    image_shape = X_test[0].shape
    X_test = (X_test.astype(np.float32) / 255.0).reshape((X_test.shape[0], image_shape[0] * image_shape[1]))
    digits = select_random_digits_from_each_class(X_test, y_test)

    model_paths = [
        os.path.join('selected_models', 'RMSprop', 'PCA_20_64_60000_784-500-200-30_0.001_0.9'),
        os.path.join('selected_models', 'RMSprop', 'LDA_50_64_10000_784-500-200-30_0.003_0.3'),
        os.path.join('selected_models', 'RMSprop', 'LE_20_64_20000_784-500-200-30_0.001_0.9'),
        os.path.join('selected_models', 'RMSprop', 'MFA_50_64_2500_784-500-200-30_0.001_0.9')
    ]

    digits_vis = digits[:10,:] * 255
    digits_vis = np.reshape(digits_vis, (10, image_shape[0], image_shape[1]))
    digits_vis = np.hstack([d for d in digits_vis])

    for i, model_path in enumerate(model_paths):
        model = ann.load_model(model_path)
        digit_reconstructed = model.predict(digits, True)
        digit_reconstructed = digit_reconstructed[:10,:] * 255
        digit_reconstructed = np.reshape(digit_reconstructed, (10, image_shape[0], image_shape[1]))
        digits_reconstructed = np.hstack([dire for dire in digit_reconstructed])
        digits_vis = np.vstack([digits_vis, digits_reconstructed])

    fig, ax = plt.subplots()
    ax.imshow(digits_vis, 'gray')
    plt.savefig(os.path.join('selected_models', 'combined.png'))
    # plt.show()

    return 0

if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))