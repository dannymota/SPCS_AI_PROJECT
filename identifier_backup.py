#import sknn
import numpy as np

from keras.models import Sequential
from keras.layers import Activation, Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import SGD

class Identify(object):

    def __init__(self, X, Y, nClasses, num):

        WEIGHTS_FILE = 'weights_and_biases%d.hd5' % num

        ### Initialize neural network ###
        #self.ReLUCNN = sknn.mlp.Convolution(type="Rectifier")
        #self.softmaxCNN = sknn.mlp.Convolution(type="Softmax")
        #self.classifier = sknn.mlp.Classifier(layers=[self.ReLUCNN, self.softmaxCNN],
        #                                      learning_rate=0.0001,
        #                                      learning_momentum=0.9,
        #                                     )

        print X.shape
        self.NeuralNetwork = Sequential()

        ## Add layers to neural network ##

        # self.NeuralNetwork.add(Convolution2D(32, 20,20, border_mode='valid', input_shape=(1, 128, 128)))
        # self.NeuralNetwork.add(Activation("relu"))
        #
        # self.NeuralNetwork.add(MaxPooling2D(pool_size=(10,10)))
        # self.NeuralNetwork.add(Dropout(0.25))
        #
        # self.NeuralNetwork.add(Flatten())
        #
        # self.NeuralNetwork.add(Dense(36))
        # self.NeuralNetwork.add(Activation("softmax"))

        self.NeuralNetwork.add(Convolution2D(32, 11, 11,
                        border_mode='valid',
                        input_shape=(1, 128, 128)))
        self.NeuralNetwork.add(Activation('relu'))
        self.NeuralNetwork.add(Convolution2D(32, 11, 11))
        self.NeuralNetwork.add(Activation('relu'))
        self.NeuralNetwork.add(MaxPooling2D(pool_size=(5, 5)))
        self.NeuralNetwork.add(Dropout(0.25))

        self.NeuralNetwork.add(Flatten())
        self.NeuralNetwork.add(Dense(128))
        self.NeuralNetwork.add(Activation('relu'))
        self.NeuralNetwork.add(Dropout(0.5))
        self.NeuralNetwork.add(Dense(36))
        self.NeuralNetwork.add(Activation('softmax'))


        optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        self.NeuralNetwork.compile(loss='categorical_crossentropy', optimizer=optimizer)

        ### Initialize input variables ###
        # X -> Feature set
        # Y -> Response variables

        self.X, self.Y = (X,Y)

        # Fit the classifier
        #self.classifier.fit(self.X, self.Y, self.W)

        self.his = self.NeuralNetwork.fit(X, Y,
                               batch_size=64,
                               validation_split=0.3,
                               nb_epoch=1000,
                               verbose=1)

        self.NeuralNetwork.save_weights(WEIGHTS_FILE)


    def predict(self, X):
        return self.Y[np.argmax(self.predict_probabilities(X), axis=1)]

    def predict_probabilities(self, X):
        return self.NeuralNetwork.predict_proba(X)

    def evaluate(self, X_test, Y_test):
        return self.NueralNetwork.evaluate(X_test, Y_test, batch_size=64)

    def getTrainingStats(self):
        return self.his
