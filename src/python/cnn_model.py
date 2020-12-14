import sys

import matplotlib.pyplot as plt
import numpy as np
import keras

from keras import layers
from sklearn.model_selection import train_test_split
from keras.callbacks import LearningRateScheduler

from utils import die


def split_dataset(dataset, testSize, randomState=13):
    trainSet, _, testSet, _ = train_test_split(dataset, dataset, test_size=testSize, random_state=randomState)
    
    return (trainSet, testSet)


def reshape(trainSet, testSet, rows, cols):
    x_train = trainSet.astype('float32') / 255.
    x_test = testSet.astype('float32') / 255.

    trainSet = np.reshape(x_train, (len(x_train), rows, cols, 1))
    testSet = np.reshape(x_test, (len(x_test), rows, cols, 1))

    return (trainSet, testSet)


class Autoencoder():

    def __init__(self, dataset, dims, epochs, batch_size, convs):
        self.dataset = dataset
        self.rows = dims[0]
        self.cols = dims[1]

        # Hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.convs = convs

        self.autoencoder = None


    def _split_dataset(self, testSize, randomState=13):
        # training data must be split into a training set and a validation set
        self.trainSet, self.testSet = split_dataset(self.dataset, testSize, randomState)


    def _reshape(self):
        # normalization
        self.trainSet, self.testSet = reshape(self.dataset, self.testSet, self.rows, self.cols)


    def __add_conv_layers(self, first_input, conv, ith_conv, dec=False):
        if (len(self.convs) % 2 == 0):
            reps = len(self.convs) // 2
        else:
            reps = (len(self.convs) - 1) // 2
        for _ in range(0, reps):
            name = str(ith_conv)
            if (dec == False):
                name += 'e'
            else:
                name += 'd'
            if ((ith_conv == 0) and (dec == False)):
                conv = layers.Conv2D(self.convs[ith_conv][0], kernel_size=self.convs[ith_conv][1], activation='relu', kernel_initializer='he_uniform', padding='same', name='conv'+name)(first_input)
            elif ((ith_conv == len(self.convs) - 1) and (dec == True)):
                conv = layers.Conv2D(self.convs[ith_conv][0], kernel_size=self.convs[ith_conv][1], activation='relu', kernel_initializer='he_uniform', padding='same', name='conv'+name)(first_input)
            else:
                conv = layers.Conv2D(self.convs[ith_conv][0], kernel_size=self.convs[ith_conv][1], activation='relu', kernel_initializer='he_uniform', padding='same', name='conv'+name)(conv)
            conv = layers.BatchNormalization(name='batch'+name)(conv)
            if (dec == False):
                ith_conv += 1
            else:
                ith_conv -= 1
                
        return (conv, ith_conv)


    def encoder(self, input_img):
        ith_conv = 0
        if (len(self.convs) == 1):
            conv = layers.Conv2D(self.convs[ith_conv][0], kernel_size=self.convs[ith_conv][1], activation='relu', kernel_initializer='he_uniform', padding='same', name='conv0e')(input_img)
            conv = layers.BatchNormalization(name='batch0e')(conv)
            conv = layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='pool1')(conv)
            conv = layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='pool2')(conv)
        else:
            conv, ith_conv = self.__add_conv_layers(input_img, None, ith_conv)
            conv = layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='pool1')(conv)

            conv, ith_conv = self.__add_conv_layers(None, conv, ith_conv)
            conv = layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='pool2')(conv)

            if (len(self.convs) % 2 != 0):
                conv = layers.Conv2D(self.convs[ith_conv][0], kernel_size=self.convs[ith_conv][1], activation='relu', kernel_initializer='he_uniform', padding='same', name='conv'+str(ith_conv)+'e')(conv)
                conv = layers.BatchNormalization(name='batch'+str(ith_conv)+'e')(conv)

        return conv


    def decoder(self, encoded):
        conv = None
        ith_conv = len(self.convs) - 1
        if (len(self.convs) == 1):
            conv = layers.Conv2D(self.convs[0][0], kernel_size=self.convs[0][1], activation='relu', kernel_initializer='he_uniform', padding='same', name='conv1d')(encoded)
            conv = layers.BatchNormalization(name='batch1d')(conv)
            conv = layers.UpSampling2D((2, 2), name='up1')(conv)
            conv = layers.UpSampling2D((2, 2), name='up2')(conv)
        else:
            if (len(self.convs) % 2 != 0):
                conv = layers.Conv2D(self.convs[0][0], kernel_size=self.convs[0][1], activation='relu', kernel_initializer='he_uniform', padding='same', name='conv'+str(ith_conv)+'d')(encoded)
                conv = layers.BatchNormalization(name='batch'+str(ith_conv)+'d')(conv)
                ith_conv = len(self.convs) - 2

            conv, ith_conv = self.__add_conv_layers(encoded, conv, ith_conv, True)
            conv = layers.UpSampling2D((2, 2), name='up1')(conv)

            conv, ith_conv = self.__add_conv_layers(None, conv, ith_conv, True)
            conv = layers.UpSampling2D((2, 2), name='up2')(conv)

        return layers.Conv2D(1, kernel_size=self.convs[0][1], activation='sigmoid', kernel_initializer='he_uniform', padding='same', name='decoded')(conv)


    def compile_model(self, input_img, decoded):
        self.autoencoder = keras.Model(input_img, decoded)
        print(self.autoencoder.summary())
        opt = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=None, decay=0.0)
        self.autoencoder.compile(optimizer=opt, loss='mean_squared_error', metrics=["accuracy"])


    def train_model(self):
        annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x, verbose=0)
        self.history = self.autoencoder.fit(self.trainSet, self.trainSet, batch_size=self.batch_size, epochs=self.epochs, validation_data=(self.testSet, self.testSet), callbacks=[annealer])


    def save_model(self, path):
        self.autoencoder.save(path)