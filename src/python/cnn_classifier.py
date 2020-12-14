from keras.models import Sequential, load_model
from keras.layers import MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import RMSprop, Adam
from keras.metrics import SparseCategoricalAccuracy
import matplotlib.pyplot as plt
import numpy as np


# default values
LEARNINGRATE = 2e-4
DROPOUT      = 0.4  # 0.5


class Classifier:

    def __init__(self, ae_model):
        self.model = Sequential()
        self.train_history = None

        # each layer name in the autoencoder ends with either 'e' or 'd'
        # 'e': encoder layer, 'd': decoder layer
        for i in range(len(ae_model.layers)):
          if ae_model.layers[i].name[-1] == "d":
              break
          self.model.add(ae_model.get_layer(index=i))

          # after each max pooling layer add a dropout layer
          if ae_model.layers[i].name[:4] == "pool":
              self.model.add(Dropout(DROPOUT))

        self.model.add(Flatten())


    def add_layers(self, fc_nodes):
        self.model.add(Dense(fc_nodes, activation="relu", name="fully_connected"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(DROPOUT))
        self.model.add(Dense(10, activation="softmax", name="output"))
        print(self.model.summary())


    def train(self, train_images, train_labels, val_images, val_labels):
        # 1st training stage: train only the weights of the fc layer, "freeze" the rest
        for l in self.model.layers:
            if l.name != "fully_connected": 
                l.trainable = False

        # compile 
        self.model.compile(loss=SparseCategoricalCrossentropy(),
                            optimizer=Adam(learning_rate=LEARNINGRATE),
                            metrics=[SparseCategoricalAccuracy()])       

        epochs1     = int(input("\n> Enter training epochs for training stage 1: "))
        minibatch1  = int(input("> Enter training batch size for training stage 1: "))
        print("\nTraining Stage 1: Training only the Fully-Connected layer's weights...")
        self.model.fit(train_images, train_labels, batch_size=minibatch1, epochs=epochs1, validation_data=(val_images, val_labels))
        print("Done!\n")

        # 2nd training stage: train the entire network
        for l in self.model.layers:
            l.trainable = True 

        # re-compile the model and repeat training
        self.model.compile(loss=SparseCategoricalCrossentropy(),
                            optimizer=Adam(learning_rate=LEARNINGRATE),
                            metrics=[SparseCategoricalAccuracy()])       

        epochs2     = int(input("> Enter training epochs for training stage 2: "))
        minibatch2  = int(input("> Enter training batch size for training stage 2: "))
        print("\nTraining Stage 2: Training the entire network...")
        self.train_history = self.model.fit(train_images, train_labels, batch_size=minibatch2, 
                                        epochs=epochs2, validation_data=(val_images, val_labels))
        print("Done!\n")

        # we use epochs and batch size of the 2nd training stage for plotting
        return (epochs2, minibatch2)


    def test(self, test_images, test_labels, size):
        y_pred1 = self.model.predict(test_images)
        y_pred2 = np.argmax(y_pred1, axis=1)

        res = self.model.evaluate(test_images, test_labels)
        print("\nClassifier Test Accuracy = {}".format(res[1]))
        print("Classifier Test Loss = {}".format(res[0]))

        return (y_pred2, res[1], res[0])
    
    
    def plot_acc_loss(self):
        fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2)
        fig.tight_layout()

        ax1.plot(self.train_history.history["sparse_categorical_accuracy"])
        ax1.plot(self.train_history.history["val_sparse_categorical_accuracy"])
        ax1.set_title("Training Curve")
        ax1.set_ylabel("accuracy")
        ax1.set_xlabel("epochs")
        ax1.legend(["train accuracy", "val accuracy"], loc="lower right")

        ax2.plot(self.train_history.history["loss"])
        ax2.plot(self.train_history.history["val_loss"])
        ax2.set_title("Training Curve") 
        ax2.set_ylabel("loss")
        ax2.set_xlabel("epochs")
        ax2.legend(["train loss", "val loss"], loc="upper right")

        fig.subplots_adjust(wspace=0.5)
        plt.show()
