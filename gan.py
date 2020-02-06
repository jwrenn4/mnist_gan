import tensorflow as tf
import numpy as np

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return {
        i : np.concatenate((x_train[(y_train == i).flatten()], x_test[(y_test == i).flatten()])).reshape(-1,28,28,1)
        for i in np.unique(y_train)
    }

def build_generator(input_dimension = 100):
    input_layer = tf.keras.layers.Input(input_dimension)
    x = tf.keras.layers.Dense(28*28, activation = 'relu')(input_layer)
    x = tf.keras.layers.Dense(28*28, activation = 'relu')(x)
    x = tf.keras.layers.Dense(28*28, activation = 'relu')(x)
    x = tf.keras.layers.Dense(28*28, activation = 'relu')(x)
    x = tf.keras.layers.Dense(28*28, activation = 'relu')(x)    
    output_layer = tf.keras.layers.Reshape((28, 28, 1))(x)
    return tf.keras.models.Model(inputs = input_layer, outputs = output_layer)

def build_discriminator():
    input_layer = tf.keras.layers.Input((28,28,1))
    x = tf.keras.layers.Conv2D(512, 2, padding = 'same')(input_layer)
    x = tf.keras.layers.Conv2D(512, 2, padding = 'same')(x)
    x = tf.keras.layers.Conv2D(512, 2, padding = 'valid')(x)
    x = tf.keras.layers.Conv2D(512, 2, padding = 'valid')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation = 'relu')(x)
    output_layer = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)
    return tf.keras.models.Model(inputs = input_layer, outputs = output_layer)

def train_gan_epoch(generator, discriminator):
    return
