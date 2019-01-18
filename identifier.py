from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD

class Identify(object):

    def __init__(self, X, Y, X_test, y_test, num):

        WEIGHT_FILE = "weights%d.hd5" % num

        N_CLASSES = 36
        N_EPOCHS = 10
        BATCH_SIZE = 128

        # input image dimensions
        N_ROWS, N_COLS = 128, 128

        # number of convolutional filters to use
        N_FILTERS = 36

        # size of pooling area for max pooling
        POOL_SIZE = 2

        # convolution kernel size
        KERNEL_SIZE = 5

# convert class vectors to binary class matrices
        Y = np_utils.to_categorical(Y, N_CLASSES)
        y_test = np_utils.to_categorical(y_test, N_CLASSES)
        self.model = Sequential()

        self.model.add(Convolution2D(N_FILTERS, KERNEL_SIZE, KERNEL_SIZE,
                                border_mode='valid',
                                input_shape=(1, N_ROWS, N_COLS)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(N_FILTERS, KERNEL_SIZE, KERNEL_SIZE))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(POOL_SIZE, POOL_SIZE)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(128))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(N_CLASSES))
        self.model.add(Activation('softmax'))

        optimize = SGD(lr=0.0001, momentum=0.9, decay=0.2)

        self.model.compile(loss='categorical_crossentropy',
                      optimizer=optimize,
                      metrics=['accuracy'])

        self.his = self.model.fit(X, Y, batch_size=BATCH_SIZE, nb_epoch=N_EPOCHS,
                  verbose=1, validation_data = (X_test, y_test))

        self.model.save_weights(WEIGHT_FILE)

    def predict(self, X):
        return self.model.predict(X)

    def predict_probabilities(self, X):
        return self.model.predict_proba(X)
    def evaluate(self, X_test, Y_test):
        return self.model.evaluate(X_test, Y_test, batch_size=64)

    def getTrainingStats(self):
        return self.his
