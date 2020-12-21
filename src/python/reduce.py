import argparse
import os.path
import sys

import matplotlib.pyplot as plt
import keras

from utils import dataset_reader, die, ask_for_hyperparameters
from cnn_model import Autoencoder, split_dataset, reshape

from keras import backend as K


# construct a custom layer to calculate the loss
class custom_variation_layer(keras.layers.Layer):

    def vae_loss(self, x, z_decoded, z_mu, z_log_sigma):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        # Reconstruction loss
        xent_loss = keras.binary_crossentropy(x, z_decoded)
        # KL divergence
        kl_loss = -5e-4 * K.mean(1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma), axis=-1)
        return K.mean(xent_loss + kl_loss)

    # adds the custom loss to the class
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x

def cnn_train_simulation():
    latent_dim, epochs, batch_size = ask_for_hyperparameters()

    autoencoder = Autoencoder(dataset, dims, epochs, batch_size, latent_dim)
    autoencoder.process_data(testSize)

    input_img = keras.Input(shape=(rows, cols, 1))
    encoded = autoencoder.encoder(input_img)
    z = keras.Lambda(autoencoder.sampling)([autoencoder.z_mu, autoencoder.z_log_sigma])
    decoder_input = keras.Input(K.int_shape(z)[1:])
    decoder = autoencoder.decoder(decoder_input)
    z_decoded = decoder(z)

    y = custom_variation_layer()([input_img, z_decoded, autoencoder.z_mu, autoencoder.z_log_sigma])
    vae = keras.Model(input_img, y)
    print(vae.summary())
    vae.compile(optimizer='rmsprop', loss=None)
    # autoencoder.compile_model(input_img, decoded)
    # autoencoder.train_model()

    # return autoencoder
    return vae

def save_model(model, path):
    model.save(path)


# def fit_pretrained(modelPath, dataset, rows, cols, testSize):
#     epochs = int(input("> Enter training epochs: "))
#     batch_size = int(input("> Enter training batch size: "))

#     trainSet, testSet = split_dataset(dataset, testSize)
#     trainSet, testSet = reshape(trainSet, testSet, rows, cols)

#     model = keras.models.load_model(modelPath)
#     print(model.summary())
#     hist = model.fit(trainSet, trainSet, batch_size=batch_size, epochs=epochs, validation_data=(testSet, testSet), verbose=2)



def menu():
    print("\n1. Do you want to repeat the experiment with other hyperparameter values?")
    print("2. Do you want to save the trained model with the latest hyperparameter values?")
    print("3. Do you want to exit the program?")
    code = input("Provide one of the above codes: ")
    
    return code



if __name__ == '__main__':

    # parsing the command line arguments with argparse module
    # --help provides instructions
    parser = argparse.ArgumentParser(description = 'Provide dataset file')
    parser.add_argument('-d', '--dataset', required = True, help = 'dataset file path')
    parser.add_argument('-q', '--queryset', required = True, help = 'queryset file path')
    parser.add_argument('-od', '--od', required = True, help = 'output dataset file')
    parser.add_argument('-oq', '--oq', required = True, help = 'output query file')
    args = vars(parser.parse_args())


    # if dataset file exists, otherwise exit
    dataset = args['dataset']
    # dataset = 'train-images-idx3-ubyte'
    if (os.path.isfile(dataset) == False):
        die("\n[+] Error: This dataset file does not exist\n\nExiting...", -1)

    queryset = args['queryset']
    if (os.path.isfile(queryset) == False):
        die("\n[+] Error: This queryset file does not exist\n\nExiting...", -1)


    # read the first 16 bytes (magic_number, number_of_images, rows, columns)
    dataset, _, rows, cols = dataset_reader(dataset)
    dims = (rows, cols)

    testSize = 0.2

    # check = input("Do you want to load a pretrained cnn model? [y/n]: ")
    # if (check == 'y'):
    #     modelPath = input("> Give the path in which the cnn model is located: ")
    #     if (os.path.isfile(modelPath) == False):
    #         die("\n[+] Error: File \"{}\" does not exist!\n".format(modelPath), -1)
    #     fit_pretrained(modelPath, dataset, rows, cols, testSize)
    # else:
    model = cnn_train_simulation()

    while (True):
        code = menu()
        if (code == '1'):
            model = cnn_train_simulation()
        elif (code == '2'):
            path = input("> Give the path where the CNN will be saved: ")
            save_model(model, path)
        elif (code == '3'):
            sys.exit(0)
        else:
            print("[+] Error: Provide only one of the above codes (1, 2, 3)")
            continue

