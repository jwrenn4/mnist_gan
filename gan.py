import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
    output_layer = tf.keras.layers.Reshape((28, 28, 1))(x)
    return tf.keras.models.Model(inputs = input_layer, outputs = output_layer)

def build_discriminator():
    input_layer = tf.keras.layers.Input((28,28,1))
    x = tf.keras.layers.Conv2D(512, 3, padding = 'same')(input_layer)
    x = tf.keras.layers.Conv2D(512, 3, padding = 'same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(8, activation = 'relu')(x)
    output_layer = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)
    return tf.keras.models.Model(inputs = input_layer, outputs = output_layer)

def train_gan_epoch(generator, discriminator, data, input_dimension = 100):

    # construce the gan
    input_layer = tf.keras.layers.Input(input_dimension)
    generator_out = generator(input_layer)
    discriminator_out = discriminator(generator_out)
    gan = tf.keras.models.Model(inputs = input_layer, outputs = discriminator_out)

    # first part, train the discriminator to identify fake images
    generator.trainable = False
    discriminator.trainable = True

    # create the data to train off of
    fake_images = generator.predict(np.random.random((data.shape[0], input_dimension)))

    real_labels = np.zeros(data.shape[0]).reshape(-1, 1) + 0.05
    fake_labels = np.ones(data.shape[0]).reshape(-1, 1) - 0.05
    training_images = np.concatenate((data, fake_images))
    training_labels = np.concatenate((real_labels, fake_labels))

    order = np.random.choice(np.arange(training_images.shape[0]), training_images.shape[0], replace = False)
    training_images = training_images[order]
    training_labels = training_labels[order]

    discriminator.compile(loss = 'binary_crossentropy', optimizer = 'adam')
    discriminator.fit(training_images, training_labels, batch_size = 16, epochs = 1)

    # now train the generator through the gan
    generator.trainable = True
    discriminator.trainable = False

    # generate noise to predict off of and labels
    random_noise = np.random.random((data.shape[0], input_dimension))
    noise_labels = np.zeros(data.shape[0]).reshape(-1, 1) + 0.05
    gan.compile(loss = 'binary_crossentropy', optimizer = 'adam')
    gan.fit(random_noise, noise_labels, batch_size = 16, epochs = 1)

if __name__ == '__main__':
    generator = build_generator()
    discriminator = build_discriminator()
    data = load_data()[0] / 256

    for i in range(10):
        epoch_num = i + 1
        train_gan_epoch(generator, discriminator, data)
        random_input = np.random.random((1, 100))
        generated_image = generator.predict(random_input).reshape((28, 28))
        plt.imsave(f'model_0_epoch_{epoch_num}.png', generated_image)
        generator.save('model_0.h5')
        
