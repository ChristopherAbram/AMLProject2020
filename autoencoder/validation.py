import tensorflow as tf
import matplotlib.pyplot as plt
import math
import autoencoder.nn as ann

def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    x = tf.reshape(x, [28*28])
    y = tf.cast(y, tf.int32)
    return x, y

def postprocess(X, image_shape=(28, 28)):
    return tf.reshape(X, image_shape) * 255

# ERROR RATE and IMPURITY
def error_rate_impurity(X_valid_encoded, X_valid, y_valid, k=18):
    errors = 0
    impurities = 0
    for i, x_enc in enumerate(X_valid_encoded):
        top_k_indices = ann.knn(x_enc, X_valid_encoded, k)
        label = y_valid[i]
        votes_against = 0
        for index in top_k_indices:
            if label != y_valid[index]:
                votes_against += 1
        if votes_against > math.ceil(k / 2):
            errors += 1
        impurities += votes_against
    error_rate = errors * 100. / X_valid.shape[0]
    impurity = impurities / (X_valid.shape[0] * k)
    return error_rate, impurity

# 2D GRAPH
def scatter_plot_2d(filepath, X_valid_encoded, X_valid, y_valid, n_classes):
    fig_, ax = plt.subplots()
    class_points_x = [[] for i in range(n_classes)]
    class_points_y = [[] for i in range(n_classes)]
    for i, (e, y) in enumerate(zip(X_valid_encoded, y_valid)):
        pp_e = postprocess(e, (1, 2))
        coord = pp_e.numpy().ravel()
        class_points_x[y].append(coord[0])
        class_points_y[y].append(coord[1])
    for label in range(n_classes):
        ax.scatter(class_points_x[label], class_points_y[label], label="%d" % label)
    plt.legend()
    plt.savefig(filepath + "/img/2d_encode.png")

def encoded_display_shape(hidden_size):
    width = math.sqrt(hidden_size)
    height = width
    if not width.is_integer():
        width = hidden_size
        height = 1
    else:
        width = int(width)
        height = int(height)
    return width, height

def plot_table(filepath, X_valid, X_valid_predicted, X_valid_encoded, 
    image_shape, hidden_shape, shape):

    fig, axs = plt.subplots(shape[0], shape[1] * 3, figsize=(10, 10))
    axs_ = axs.ravel()
    for i, (x, xp, xen) in enumerate(zip(X_valid, X_valid_predicted, X_valid_encoded)):
        img_org = postprocess(x, image_shape)
        img_pred = postprocess(xp, image_shape)
        img_encoded = postprocess(xen, (hidden_shape[1], hidden_shape[0]))
        axs_[i * 3].axis('off')
        axs_[i * 3 + 1].axis('off')
        axs_[i * 3 + 2].axis('off')
        axs_[i * 3].imshow(img_org, cmap='gray')
        axs_[i * 3 + 1].imshow(img_pred, cmap='gray')
        axs_[i * 3 + 2].imshow(img_encoded, cmap='gray')
    plt.savefig(filepath + "/img/predict.png")

