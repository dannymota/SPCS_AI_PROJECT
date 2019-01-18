import numpy as np
import scipy.misc as spm

import loadData as ld

for i in range(3, 11):
    num = str(i)

    data = np.load('data/data' + num + '.npy')


    newData = np.empty([0, 32, 32])

    for a in range(data.shape[0]):
        print a
        newData = np.concatenate((newData, (spm.imresize(data[a], (32,32))).reshape(1, 32, 32)), axis = 0)


    np.save('newdata' + num + '.npy', newData)
    
    
    