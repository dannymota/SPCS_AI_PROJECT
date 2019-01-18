import numpy as np
from random import randint
import matplotlib.pyplot as plt


def createTestData(X, y, word):
    img_word = []
    word = str(word)

    for char in range(len(word)):
        if(word[char] == ' '):
            img_word.append(-1)
        else:
            indices = [i for i, x in enumerate(y) if x == word[char]]
            if len(indices) == 0:
                raise ReferenceError('No image of text: "%s"' % word[char])
            img_word.append(indices[randint(0, len(indices)-1)])
            del indices


    img = np.empty([128,0])
    for i in range(len(word)):
        if img_word[i] == -1:
            tmp = np.zeros((128,128))
            tmp.fill(255)
            img = np.concatenate((img, tmp), axis=1)
            del tmp
        else:
            img = np.concatenate((img, X[img_word[i]]), axis=1)

    return img



def mserSegmentify(img): ## experimental
    import cv2
    img = cv2.imread('image.jpg', 0);
    vis = img.copy()
    mser = cv2.MSER_create()
    regions = mser.detectRegions(img, None)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(vis, hulls, 4, (0, 255, 0))
    cv2.imshow('img', vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def lineSegmentify(img):
    img = img.T
    actualImg = np.empty([128, 0])
    actualImgs = np.empty([0,128,128])
    oldPresent = None
    for row in xrange(img.shape[0]): # len

        imagePresent = False
        for col in xrange(img.shape[1]): #wid
            if img[row][col] < 200:
                imagePresent = True # true if black pixel is found



        if(oldPresent == 1 and imagePresent == 0) or (row == (image.shape[0] - 1)):

            padLeft = int(np.floor((128-actualImg.shape[1])/2.0))
            padRight = int(np.ceil((128-actualImg.shape[1])/2.0))
            vec128 = np.zeros((128,1))
            vec128.fill(255)

            for i in xrange(padLeft):
                actualImg = np.concatenate((vec128, actualImg), axis=1)

            for i in xrange(padRight):
                actualImg = np.concatenate((actualImg, vec128), axis=1)

            actualImgs = np.concatenate((actualImgs, actualImg.reshape(1,128,128)), axis = 0)
            actualImg = np.empty([128, 0])
        if imagePresent == True:
            actualImg = np.concatenate((actualImg, img[row].reshape(128,1)), axis = 1)




        oldPresent = imagePresent


    return actualImgs

def overfitSegmentify(img):
    print img.shape
    for row in range(img.shape[1]): #length
        for col in range(img.shape[0]): # width
            if(img[col][row] < 200):
                img[col][row] = 0.5
            else:
                pass
    return img

def segmentify(img, algorithm='line'):
    if(algorithm=='line'):
        return lineSegmentify(img)
    elif(algorithm=='overfit'):
        return overfitSegmentify(img)
    elif(algorithm=='mser'):
        return mserSegmentify(img)
    else:
        raise ValueError('Please specify a valid algorithm')



#THRESHOLD = 200
#
#def segmentify2(image):
#
#    image = image.T
#    imageStarted = False
#
#    parts = []
#    start = None
#
#    for i in xrange(image.shape[0]):
#
#        if np.any(image[i] < THRESHOLD) and not imageStarted:
#
#            imageStarted = True
#            start = i
#
#        elif imageStarted and not np.any(image[i] < THRESHOLD):
#            imageStarted = False
#
#            seperated = image[start:i].T
#            temp = np.zeros((128, 128)) + 255
#
#            temp[:seperated.shape[0],:seperated.shape[1]] = seperated
#            parts.append(temp)
#
#    return np.array(parts)
#
#


if __name__ == '__main__':
    # load data
    import loadData as lD
    X, y = lD.loadNISTSD19(amt_batches=1)

    # create sample data to perform image segmentation
    img = createTestData(X, y, '4chan org h')
    del X, y

    # perform image segmentation, return 3D array
    img = segmentify(img)

    # show image
    for i in range(img.shape[0]):
        plt.imshow(img[i])
        plt.show()
