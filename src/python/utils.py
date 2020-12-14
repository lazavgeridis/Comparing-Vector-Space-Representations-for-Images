import struct
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from os.path import exists


# assuming MNIST dataset file format (big endian)
def dataset_reader(path):
    f = open(path, 'rb')
    magic, size, rows, cols = struct.unpack(">IIII", f.read(16))      # reads 4 integers (16 bytes) that are in big-endian format
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>')) # we know that each pixel takes values in [0, 255]
    data = data.reshape((size, rows, cols))
    f.close()

    return (data, size, rows, cols)


# assuming MNIST labels file format (big endian)
def labels_reader(path):
    f = open(path, 'rb')
    magic, size = struct.unpack(">II", f.read(8))
    labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    f.close()

    return labels


def die(error_message, error_code):
    print(error_message, file=sys.stderr)
    sys.exit(error_code)


def ask_for_hyperparameters():
    epochs = int(input("> Enter training epochs: "))
    batch_size = int(input("> Enter training batch size: "))
    n_convs = int(input("> Enter the number of convolutional layers: "))
    convs = []
    for i in range(n_convs):
        filts = int(input("> Enter the number of filters for convolutional layer {}: ".format(i+1)))
        size = int(input("> Enter the filter size for convolutional layer {}: ".format(i+1)))
        convs.append((filts, size))
    
    return (epochs, batch_size, convs)


def classification_parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", help="path to file containing the training set samples")
    parser.add_argument("-dl", help="path to file containing the training set labels")
    parser.add_argument("-t", help="path to file containing the test set samples")
    parser.add_argument("-tl", help="path to file containing the test set labels")
    parser.add_argument("-model", help="path to autoencoder model")
    args = parser.parse_args()
     
    # check if the file arguments exist
    if not exists(args.d):
        die("\nFile \"{}\" does not exist!\n".format(args.d), -1)
     
    if not exists(args.dl):
        die("\nFile \"{}\" does not exist!\n".format(args.dl), -1)
     
    if not exists(args.t):
        die("\nFile \"{}\" does not exist!\n".format(args.t), -1)
     
    if not exists(args.tl):
        die("\nFile \"{}\" does not exist!\n".format(args.tl), -1)
     
    if not exists(args.model):
        die("\nFile \"{}\" does not exist!\n".format(args.model), -1)

    return args


# models is a list of tuples
# each tuple : (model, fully_connected layer nodes, epochs trained, batch size used, test accuracy, test loss)
def plot_nn2(models, test_labels, predictions):
    # in case user trained only one cnn model
    # plot accuracy and loss wrt #epochs only for this model
    if len(models) == 1:
        m, _, _, _, _, _ = models[0] 
        print(classification_report(predictions[0], test_labels, digits=3))
        m.plot_acc_loss()
        return 

    nodes = set()
    epoch = set()
    batch = set()
    acc = []
    val_acc = []
    loss = []
    val_loss = []

    for i, item in enumerate(models):
        model, fc_nodes, epochs, batch_size, _, _ = item
        nodes.add(fc_nodes)
        epoch.add(epochs)
        batch.add(batch_size)
        acc.append(model.train_history.history["sparse_categorical_accuracy"][epochs - 1])
        val_acc.append(model.train_history.history["val_sparse_categorical_accuracy"][epochs - 1])
        loss.append(model.train_history.history["loss"][epochs - 1])
        val_loss.append(model.train_history.history["val_loss"][epochs - 1])

        print("\nCNN Classifier Model {}: Fully-Connected nodes={}, Epochs={}, Batch Size={}".format(i + 1, fc_nodes, epochs, batch_size))
        print(classification_report(predictions[i], test_labels, digits=3))

    # in order to plot accuracy and loss, we need only one variable hyperparameter i.e one of fc_nodes, epochs, batch size
    # if the other 2 hyperparameters were kept fixed during training, then we can successfully plot
    # if the other 2 hyperparameters were not kept fixed, then loss and accuraracy cannot be plotted
    variable = 0
    if len(nodes) > 1:
        variable += 1
    if len(epoch) > 1:
        variable += 1
    if len(batch) > 1:
        variable += 1

    if variable != 1:
        die("\nCould not plot accuracy and loss graphs!\n", -3)

    # plot accuracy, loss with respect to ...
    fig2, ((ax3, ax4)) = plt.subplots(nrows=1, ncols=2)
    fig2.tight_layout()

    if len(nodes) > 1:
        l = list(nodes)
        variable = "fc nodes"
    elif len(epoch) > 1:
        l = list(epoch)
        variable = "epochs"
    else:
        l = list(batch)
        variable = "batch size"

    ax3.plot(l, acc)
    ax3.plot(l, val_acc)
    ax3.set_title("Training Curve")
    ax3.set_ylabel("accuracy")
    ax3.set_xlabel("{}".format(variable))
    ax3.legend(["train accuracy", "val accuracy"], loc="lower right")

    ax4.plot(l, loss)
    ax4.plot(l, val_loss)
    ax4.set_title("Training Curve") 
    ax4.set_ylabel("loss")
    ax4.set_xlabel("{}".format(variable))
    ax4.legend(["train loss", "val loss"], loc="upper right")

    fig2.subplots_adjust(wspace=0.5)
    plt.show()


def previous_experiment(experiment):
    _, fc_nodes, epochs, batch_size, _, _ = experiment
    print("\nHyperparams used in the previous experiment: fc_nodes={}, epochs={}, batch size={}\n".format(fc_nodes, epochs, batch_size))


def show_models(models):
    for i, item in enumerate(models):
        _, fc_nodes, epochs, batch_size, accuracy, loss = item
        print("\nCNN Model {}: Fully-Connected Layer's nodes={}   Epochs={}   Batch Size={}   Accuracy={:5.3f}   Loss={:5.3f}"
                                                            .format(i + 1, fc_nodes, epochs, batch_size, accuracy, loss))
        print("-" * 105)


def visualize_predictions(test_images, test_labels, size, x_dim, y_dim, pred_labels):

    true  = [index for index in range(size) if pred_labels[index] == test_labels[index]]
    false = [index for index in range(size) if pred_labels[index] != test_labels[index]] 

    print("\nFound {} correct labels".format(len(true)))
    print("Found {} incorrect labels\n".format(len(false)))

    cnt = 10
    
    # show predicted = true
    for i in range(cnt):
        plt.title("Predicted={}, True={}".format(pred_labels[true[i]], test_labels[true[i]]))
        imgtrue = test_images[true[i]].reshape(x_dim, y_dim)
        plt.imshow(imgtrue, cmap='gray')
        plt.show()

   # show predicted != true 
    for i in range(cnt):
        plt.title("Predicted={}, True={}".format(pred_labels[false[i]], test_labels[false[i]]))
        imgfalse = test_images[false[i]].reshape(x_dim, y_dim)
        plt.imshow(imgfalse, cmap='gray')
        plt.show()
