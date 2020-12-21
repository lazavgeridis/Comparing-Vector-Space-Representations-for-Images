import argparse
import os.path
import sys

import matplotlib.pyplot as plt
import keras

from keras import backend as K
from utils import dataset_reader, die, ask_for_hyperparameters, write_output_data
from cnn_model import Autoencoder, process_data, train_model



def cnn_train_simulation(trainSet, testSet):
    latent_dim, epochs, batch_size = ask_for_hyperparameters()

    check = input("Do you want to load a pretrained cnn model? [y/n]: ")
    if (check == 'y'):
        modelPath = input("> Give the path in which the cnn model is located: ")
        if (os.path.isfile(modelPath) == False):
            die("\n[+] Error: File \"{}\" does not exist!\n".format(modelPath), -1)
        model = keras.models.load_model(modelPath)
    else:
        autoencoder = Autoencoder(trainSet, testSet, epochs, batch_size, latent_dim)
        input_img = keras.Input(shape=(rows, cols, 1))
        encoded = autoencoder.encoder(input_img)
        decoded = autoencoder.decoder(encoded)
        model = autoencoder.compile_model(input_img, decoded)

    train_model(model, trainSet, testSet, epochs, batch_size)
    
    return model


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
    queryset, _, rows, cols = dataset_reader(queryset)
    testSize = 0.2

    trainSet, testSet = process_data(dataset, rows, cols, testSize)

    model = cnn_train_simulation(trainSet, testSet)

    half_model = keras.Model(inputs=model.layers[0].input, outputs=model.get_layer("bottleneck").output)

    bottleneck_dataset_outputs = half_model.predict(dataset)
    bottleneck_query_outputs = half_model.predict(queryset)

    write_output_data(args['od'], bottleneck_dataset_outputs)
    write_output_data(args['oq'], bottleneck_query_outputs)

