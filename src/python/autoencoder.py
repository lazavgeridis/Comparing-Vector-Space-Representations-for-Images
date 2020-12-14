import argparse
import os.path
import sys

import matplotlib.pyplot as plt
import keras

from utils import dataset_reader, die, ask_for_hyperparameters
from cnn_model import Autoencoder, split_dataset, reshape


# plot loss for pre-trained model only, no need to plot other hyperparameters
def plot_loss(hist):
    # plot loss
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'test'], loc='upper left')
    name = input("> Give the path where the plot will be saved: ")
    plt.savefig(name)
    plt.show()



def plot_hyperparams(models):
    conv_layers = set()
    n_epochs = set()
    batch = set()
    loss = []
    val_loss = []

    for i, item in enumerate(models):
        model, epochs, batch_size, convs = item
        conv_layers.add(len(convs))
        n_epochs.add(epochs)
        batch.add(batch_size)
        loss.append(model.history.history["loss"][epochs-1])
        val_loss.append(model.history.history["val_loss"][epochs-1])

        print("\nCNN Autoencoder Model {}: Convolution Layers={}, Epochs={}, Batch Size={}".format(i + 1, len(convs), epochs, batch_size))


    # in order to plot accuracy and loss, we need only one variable hyperparameter i.e one of fc_nodes, epochs, batch size
    # if the other 2 hyperparameters were kept fixed during training, then we can successfully plot
    # if the other 2 hyperparameters were not kept fixed, then loss and accuraracy cannot be plotted
    variable = 0
    if len(conv_layers) > 1:
        variable += 1
    if len(n_epochs) > 1:
        variable += 1
    if len(batch) > 1:
        variable += 1

    if variable != 1:
        die("\nCould not plot loss graphs!\n", -1)

    if len(conv_layers) > 1:
        l = list(conv_layers)
        variable = "Convolutional Layers"
    elif len(n_epochs) > 1:
        l = list(n_epochs)
        variable = "Epochs"
    else:
        l = list(batch)
        variable = "Batch Size"

    plt.plot(l, loss)
    plt.plot(l, val_loss)
    plt.title("Training Curve") 
    plt.ylabel("Loss")
    plt.xlabel("{}".format(variable))
    plt.legend(["train_loss", "val_loss"], loc="upper left")
    name = input("> Give the path where the plot will be saved: ")
    plt.savefig(name)
    plt.show()



def cnn_train_simulation():
    epochs, batch_size, convs = ask_for_hyperparameters()

    autoencoder = Autoencoder(dataset, dims, epochs, batch_size, convs)
    autoencoder._split_dataset(testSize)
    autoencoder._reshape()

    input_img = keras.Input(shape=(rows, cols, 1))
    encoded = autoencoder.encoder(input_img)
    decoded = autoencoder.decoder(encoded)
    autoencoder.compile_model(input_img, decoded)
    autoencoder.train_model()

    return autoencoder



def fit_pretrained(modelPath, dataset, rows, cols, testSize):
    epochs = int(input("> Enter training epochs: "))
    batch_size = int(input("> Enter training batch size: "))

    trainSet, testSet = split_dataset(dataset, testSize)
    trainSet, testSet = reshape(trainSet, testSet, rows, cols)

    model = keras.models.load_model(modelPath)
    print(model.summary())
    hist = model.fit(trainSet, trainSet, batch_size=batch_size, epochs=epochs, validation_data=(testSet, testSet), verbose=2)
    plot_loss(hist)



def menu():
    print("\n1. Do you want to repeat the experiment with other hyperparameter values?")
    print("2. Do you want the error graphs to be displayed as to the values ​​of the hyperparameters for the performed experiments?")
    print("3. Do you want to save the trained model with the latest hyperparameter values?")
    print("4. Do you want to exit the program?")
    code = input("Provide one of the above codes: ")
    
    return code



if __name__ == '__main__':

    # parsing the command line arguments with argparse module
    # --help provides instructions
    parser = argparse.ArgumentParser(description = 'Provide dataset file')
    parser.add_argument('-d', '--dataset', required = True, help = 'dataset file path')
    args = vars(parser.parse_args())

    # if dataset file exists, otherwise exit
    dataset = args['dataset']
    # dataset = 'train-images-idx3-ubyte'
    if (os.path.isfile(dataset) == False):
        die("\n[+] Error: This dataset file does not exist\n\nExiting..", -1)


    # read the first 16 bytes (magic_number, number_of_images, rows, columns)
    dataset, _, rows, cols = dataset_reader(dataset)
    dims = (rows, cols)

    testSize = 0.2

    models = []

    check = input("Do you want to load a pretrained cnn model? [y/n]: ")
    if (check == 'y'):
        modelPath = input("> Give the path in which the cnn model is located: ")
        if (os.path.isfile(modelPath) == False):
            die("\n[+] Error: File \"{}\" does not exist!\n".format(modelPath), -1)
        fit_pretrained(modelPath, dataset, rows, cols, testSize)
    else:
        autoencoder = cnn_train_simulation()
        models.append( (autoencoder, autoencoder.epochs, autoencoder.batch_size, autoencoder.convs) )

    while (True):
        code = menu()
        if (code == '1'):
            autoencoder = cnn_train_simulation()
            models.append( (autoencoder, autoencoder.epochs, autoencoder.batch_size, autoencoder.convs) )
        if (code == '2'):
            if (len(models) == 1):
                plot_loss(autoencoder.history)
            else:
                plot_hyperparams(models)
        elif (code == '3'):
            path = input("> Give the path where the CNN will be saved: ")
            autoencoder.save_model(path)
        elif (code == '4'):
            sys.exit(0)
        else:
            print("[+] Error: Provide only one of the above codes (1, 2, 3, 4)")
            continue

