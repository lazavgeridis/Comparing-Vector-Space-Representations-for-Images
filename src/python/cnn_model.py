import sys

import matplotlib.pyplot as plt
import numpy as np
import keras

from keras import layers
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.callbacks import LearningRateScheduler



def process_data(dataset, rows, cols, testSize, randomState=13):
    trainSet, _, testSet, _ = train_test_split(dataset, dataset, test_size=testSize, random_state=randomState)

    x_train = trainSet.astype('float32') / 255.
    x_test = testSet.astype('float32') / 255.

    trainSet = np.reshape(x_train, (len(x_train), rows, cols, 1))
    testSet = np.reshape(x_test, (len(x_test), rows, cols, 1))

    return (trainSet, testSet)


def train_model(model, trainSet, testSet, epochs, batch_size):
    annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x, verbose=0)
    model.fit(trainSet, trainSet, batch_size=batch_size, epochs=epochs, validation_data=(testSet, testSet), callbacks=[annealer])



class Autoencoder():

    def __init__(self, trainSet, testSet, epochs, batch_size, latent_dim=10):

        self.trainSet, self.testSet = trainSet, testSet
        # Hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.latent_dim = latent_dim

        self.autoencoder = None


    def encoder(self, input_img):
        conv = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(input_img)
        conv = layers.BatchNormalization()(conv)
        conv = layers.MaxPool2D(pool_size=(2, 2), padding = 'same')(conv)
        conv = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(conv)
        conv = layers.BatchNormalization()(conv)
        conv = layers.MaxPool2D(pool_size=(2, 2), padding= 'same')(conv)
        conv = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(conv)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')(conv)
        conv = layers.BatchNormalization()(conv)
        
        self.shape_before_flattening = K.int_shape(conv)

        conv = layers.Flatten()(conv)
        conv = layers.Dense(self.latent_dim, activation='relu', name='bottleneck')(conv)

        return conv


    def decoder(self, decoder_input):
        conv = layers.Dense(np.prod(self.shape_before_flattening[1:]), activation='relu')(decoder_input)
        conv = layers.Reshape(self.shape_before_flattening[1:])(conv)
        conv = layers.Conv2DTranspose(256, kernel_size=(3, 3), padding='same', activation='relu')(conv)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Conv2DTranspose(128, kernel_size=3, padding='same', activation='relu')(conv)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Conv2DTranspose(64, kernel_size=3, padding='same', activation='relu')(conv)
        conv = layers.BatchNormalization()(conv)
        conv = layers.UpSampling2D(size=(2, 2))(conv)
        conv = layers.Conv2DTranspose(32, kernel_size=3, padding='same', activation='relu')(conv)
        conv = layers.BatchNormalization()(conv)
        conv = layers.UpSampling2D(size=(2, 2))(conv)
        
        return layers.Conv2D(1, kernel_size=3, padding='same', activation='sigmoid')(conv)



    def compile_model(self, input_img, decoded):
        self.autoencoder = keras.Model(input_img, decoded)
        print(self.autoencoder.summary())
        opt = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=None, decay=0.0)
        self.autoencoder.compile(optimizer=opt, loss='mean_squared_error', metrics=["accuracy"])

        return self.autoencoder


    def save_model(self, path):
        self.autoencoder.save(path)