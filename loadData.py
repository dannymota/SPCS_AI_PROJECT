import numpy as np

def loadNISTSD19(amt_batches=2):
    X = np.empty([0,32,32])
    y = np.array([]).astype('|S1')
    for i in range(1, amt_batches+1):
        X = np.concatenate((X, np.load('data/newdata'+str(i)+'.npy')), axis = 0)
        y = np.concatenate((y, np.load('data/label'+str(i)+'.npy')))
    
    return X, np.char.lower(y)
    

def convertASCII(y):
    trade = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6': 6, '7':7, '8':8, '9':9, 'a':10, 'b':11, 'c':12, 'd': 13, 'e':14, 'f':15, 'g':16, 'h':17, 'i':18, 'j':19, 'k':20, 'l': 21, 'm':22, 'n':23, 'o':24, 'p':25, 'q':26, 'r':27, 's':28, 't':29, 'u':30, 'v':31, 'w':32, 'x':33, 'y':34, 'z':35}
    for i in range(len(y)):
        y[i] = trade[y[i]]
#    print y
    return y.astype('int32')

def convertOutput(y):
    trade = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f', 16: 'g', 17: 'h', 18: 'i', 19: 'j', 20: 'k', 21: 'l', 22: 'm', 23: 'n', 24: 'o', 25: 'p', 26: 'q', 27: 'r', 28: 's', 29: 't', 30: 'u', 31: 'v', 32: 'w', 33: 'x', 34: 'y', 35: 'z'}
    for i in range(len(y)):
        y[i] = trade(y[i])
    return y.astype(str)


def splitData(X, y, val=False, dimensions = 3, ascii = True, pcnt_train=0.7):
    amt_split = int(X.shape[0]*pcnt_train)

    X_train = X[:amt_split]
    y_train = y[:amt_split]

    X_tmp = X[amt_split:]
    y_tmp = y[amt_split:]


    if(val):
        tmp_split = int(X_tmp.shape[0]*float(0.50))

        X_val = X_tmp[:tmp_split]
        y_val = y_tmp[:tmp_split]

        X_test = X_tmp[tmp_split:]
        y_test = y_tmp[tmp_split:]

        del X_tmp, y_tmp
        if dimensions == 3:
            if ascii == True:
                return X_train, convertASCII(y_train.astype(str)), X_val, convertASCII(y_val.astype(str)), X_test, convertASCII(y_test.astype(str))
            else:
                return X_train, y_train.astype(str), X_val, y_val.astype(str), X_test, y_test.astype(str)
        elif dimensions == 4:
            if ascii == True:
                return X_train.reshape(X_train.shape[0], 1, 32, 32).astype('int32'), convertASCII(y_train.astype(str)), X_val.reshape(X_val.shape[0], 1, 32, 32).astype('int32'), convertASCII(y_val.astype(str)), X_test.reshape(X_test.shape[0], 1, 32, 32).astype(float), convertASCII(y_test.astype(str))
            else:
                return X_train.reshape(X_train.shape[0], 1, 32, 32).astype(float), y_train.astype(str), X_val.reshape(X_val.shape[0], 1, 32, 32).astype(float), y_val.astype(str), X_test.reshape(X_test.shape[0], 1, 32, 32).astype(float), y_test.astype(str)
        else:
            raise ValueError('dimensions must be 3 or 4')
    else:
        if dimensions == 3:
            if ascii == True:
                return X_train.astype('int32'), convertASCII(y_train), X_tmp.astype('int32'), convertASCII(y_tmp)
            else:
                return X_train, y_train.astype(str), X_tmp, y_tmp.astype(str)
        elif dimensions == 4:
            if ascii == True:
                return X_train.reshape(X_train.shape[0], 1, 32, 32).astype('int32'), convertASCII(y_train.astype(str)), X_tmp.reshape(X_tmp.shape[0], 1, 32, 32).astype('int32'), convertASCII(y_tmp.astype(str))
            else:
                return X_train.reshape(X_train.shape[0], 1, 32, 32).astype(float), y_train.astype(str), X_tmp.reshape(X_tmp.shape[0], 1, 32, 32).astype(float), y_tmp.astype(str)
        else:
            raise ValueError('dimensions must be 3 or 4')
