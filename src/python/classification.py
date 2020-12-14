from sklearn.model_selection import train_test_split
from utils import *
from cnn_classifier import *


CHANNELS = 1


def main():
    args = classification_parseargs()
    
    # read files and store the data
    trainset, _, x_dim, y_dim = dataset_reader(args.d)
    trainlabels = labels_reader(args.dl)
    testset, testset_size, _, _ = dataset_reader(args.t)
    testlabels = labels_reader(args.tl)
    
    # rescale pixels in [0, 1]
    trainset = trainset / 255.0
    testset  = testset / 255.0

    # reshape the data into tensors of shape (size, rows, cols, inchannel)
    trainset = trainset.reshape(-1, x_dim, y_dim, CHANNELS)
    testset  = testset.reshape(-1, x_dim, y_dim, CHANNELS)

    # reserve some training samples for validation
    x_train, x_val, y_train, y_val = train_test_split(trainset, trainlabels, test_size=0.1)

    # Load AutoEncoder model from part A
    ae = load_model(args.model)

    # Lists
    models = []
    predictions = []

    # Construct, train, and evaluate cnn model(s)
    while True:
        if models:
            previous_experiment(models[-1])

        fc_nodes = int(input("> Enter number of nodes in fully-connected layer: "))

        classifier = Classifier(ae)
        classifier.add_layers(fc_nodes)

        epochs, batch_size = classifier.train(x_train, y_train, x_val, y_val)
        pred, acc, loss = classifier.test(testset, testlabels, testset_size)
        models.append( (classifier, fc_nodes, epochs, batch_size, acc, loss) )
        predictions.append(pred)

        print("""\nTraining and evaluating the model was completed. You now have the following options:
                1. Repeat the training process with different number of: nodes in fully-connected layer or epochs or batch size
                2. Plot accuracy and loss graphs with respect to the hyperparameters and print evaluation scores (precision, recall, f1)
                3. Visualize model's predictions
                """)
        
        option = int(input("> Enter the corresponding number: "))
        if option < 1 or option > 3:
            die("\nInvalid option!\n", -2)
        if option != 1:
            break


    # plot accuracy, loss
    if option == 2:
        plot_nn2(models, testlabels, predictions)


    # visualize predictions
    else:
        # display trained models first 
        show_models(models)
        model_ind = int(input("\n> Enter the number corresponding to the model you want to visualize label predictions for: "))
        if model_ind < 1 or model_ind > len(models):
            die("\nInvalid option!\n", -2)

        model_ind -= 1

        # visualize some mnist images with their predicted labels
        visualize_predictions(testset, testlabels, testset_size, x_dim, y_dim, predictions[model_ind])
        

if __name__ == '__main__':
    main()
