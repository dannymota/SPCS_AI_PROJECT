import numpy as np
from skimage.filters import sobel
from skimage.morphology import watershed
from skimage.feature import blob_doh
from scipy import ndimage as ndi

class Blobs(object):

    def __init__(self):

        self.nBlobs = 0
        self.image = None
        self.blobs = None

    def setImage(self, img):

        if len(img.shape) > 2:
            img = self.rgb2gray(img)
        self.image = img

    def getImage(self):
        return self.image

    def getNBlobs(self):
        return self.nBlobs

    def getBlobs(self):
        return self.blobs

    def detectBlobs(self):

        markers = np.zeros_like(self.image)
        elevation_map = sobel(self.img)

        markers[np.where(self.img < 30)] = 1
        markers[np.where(self.img > 150)] = 2

        segmentation = watershed(elevation_map, markers)
        segmentation = ndi.binary_fill_holes(segmentation - 1)

        self.blobs = blob_doh(segmentation)
        self.nBlobs = self.blobs.shape[0]

    #### UTILITY FUNCTIONS ####
    def rgb2gray(self,rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
