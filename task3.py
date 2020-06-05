import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from tqdm import tqdm_notebook as tqdm

def combineLabel(Y1, Y2):
    '''
    Combines labels from Ymin and Ymax.

    This not really needed, but it was needed for an experiment.

    '''

    Y = np.zeros(Y1.size)

    for i in range(Y1.size):

        if Y1[i] == 1:

            Y[i] = 1

        if Y2[i] == 1:

            Y[i] = 2

    return Y


def setUpTensor(X, Ymin, Ymax, minibatch, sequenceLength):
    '''
    The function is setting up tensor for CNN.
    Input:

    X -dataset
    Ymin - minima vector
    Ymax - maxima vector
    minibatch - minibatch size
    sequenceLength - the length of the analysed sequence

    Output:

    A tensor of X which is minibatch x N
    A tensor of Ymin and Ymax which is minibatch x N

    '''


    #Initialise the arrays of the required size
    Xmini = np.zeros([minibatch, sequenceLength])
    Yminmini = np.zeros([minibatch, sequenceLength])
    Ymaxmini = np.zeros([minibatch, sequenceLength])

    #Combine both Ymin and Ymax
    Y = combineLabel(Ymin, Ymax)



    for j in range(minibatch):

        #Pick the random point in the dataset and set up required length
        i = random.randint(0, X.size - sequenceLength)

        Xmini[j] = X[i:i+sequenceLength]
        Ymini[j] = Y[i:i+sequenceLength]

    #return torch.from_numpy(Xmini.astype(np.float32)), torch.from_numpy(Yminmini.astype(np.long)), torch.from_numpy(Ymaxmini.astype(np.long))
    return torch.from_numpy(Xmini.astype(np.float32)), torch.from_numpy(Ymini.astype(np.long))
