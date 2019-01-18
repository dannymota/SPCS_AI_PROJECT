'''
Network #1: NIST characters: Will recognize a-z, A-Z, 0-9
            in images. To be trained on 4 batches of NIST training data

Network #2: Object segmentation: Will attempt to segregate the objects
            in an image. No training required.

Network #3: Image type recognitiion: Will recognize whether an image is
            of multiple objects, a number or a number in word form. To be
            trained on a mix of data.

            1. Image segmentation training data. Must recognize whether the image
               is for multiple object recognition

            2. MNIST-style images of numbers

            3. Concatenation of various NIST characters (a-z, A-Z) where the
               words formed from these chars are numbers in word form (one, two etc.)

Output layer: Takes output from network #1, #2 and creates a mapping in a dictionary
                of the form {number:[list of different forms of the number, image, word, etc.]}
                Outputs guess and related items.
'''


import numpy as np
import matplotlib.pyplot as plt

from identifier import Identify
from blobs import Blobs
from imageProc import segmentify
import loadData as ld

import cPickle as pickle
import os

from imageProc import segmentify
#
# N_CLASSES = 36
#
SAVED_STATE_FILE = "saved_state"

IS_NOT_OBJECT_THRESHOLD = 0.5


if not os.path.isfile(SAVED_STATE_FILE):
    print 'Loading Data'
    X, y = ld.loadNISTSD19(amt_batches = 1)
    X_train, y_train, X_test, y_test = ld.splitData(X, y, dimensions = 4)

    # print X_train.shape, y_train.shape, X_test.shape, y_test.shape

################################################
######## INITIALIZE THE NEURAL NETWORKS ########
################################################

# Identifies characters in an image (Trained on NIST dataset)
    print 'Training Network'
    char_identifier = Identify(X_train, y_train, X_test, y_test, 1)

    print
else:

    with open(SAVED_STATE_FILE) as f:
        saved_state = pickle.load(f)

    char_identifier = saved_state['charNeural']


# Attempts to identify number of objects in an image.
object_counter = Blobs()


################################################
########## INITIALIZATION COMPLETE #############
################################################






def test():

    raise NotImplemented


def mapAns(theMap, answer, image):

    if isinstance(str, answer):

        for key in theMap:

            if answer in theMap[key]:

                theMap[key].add(image)

                return theMap[key]
    else:

        theMap[answer].add(image)
        return theMap[answer]


def main():

    numNames = ('ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE')

    if not os.path.isfile(SAVED_STATE_FILE):
        mainMap = {i:set([i, numNames[i]]) for i in xrange(10)}
    else:
        with open(SAVED_STATE_FILE) as f:
            mainMap = pickle.load(f)['theMap']

    try:

        ###############################################
        ############ LOAD FINAL DATA ##################
        ###############################################

        with open(FINAL_INPUT_FILE) as f:
            images = np.load(f)

        for image in images:

            segments = segmentify(image)

            max_prob = (char_identifier.predict_probabilities(segments[0])).max()

            if max_prob >= IS_NOT_OBJECT_THRESHOLD:

                # Image is a number / word form of a number
                ans = ''
                for segment in segments:
                    ans += char_identifier.predict(segment)

            else:

                # Image is of multiple objects

                object_counter.setImage(image)
                object_counter.detectBlobs()
                ans = object_counter.getNBlobs()

            results = mapAns(mainMap, ans, image)

            print "Your input is the same as: ",

            for result in results:

                if isinstance(np.array, result):

                    plt.imshow(result)
                    plt.show()
                else:
                    print " ", result


        with open(SAVED_STATE_FILE, "w+") as f:
            pickle.dump({'theMap':mainMap,
                         'charNeural':char_identifier
                         }, f)


    except KeyboardInterrupt:

        with open(SAVED_STATE_FILE, "w+") as f:
            pickle.dump({'theMap':mainMap,
                         'charNeural':char_identifier
                         }, f)
