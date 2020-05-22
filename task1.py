import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from tqdm import tqdm_notebook as tqdm


def generateDataset(M = 1000, T = 17, N = 1024):

    '''
    Task 1: Generate Dataset

    The initial parameters of the function allow to create dataset with average
    10 points in the 1000 points range.

    The function creates a space of points and extacts the local maxima and local
    minima points which satisfy the requirements.

    Requirements:

    - The local maxima and minima should go one after another without repetitions
    - The distance between nearest found points should be more than k = 3
    - The distance should be not less than D = T*np.std(X[:-k] - X[k:])

    Input:
    M - a big number (ex. 1000)
    T - should be tuned in order to get the required result (the specific
    number of (maxima/minima) points in the analysed range)
    N - 1024

    Output:
    X - the generated space of points
    Ymin - the binary vector of local minima points
    Ymax - the binary vector of local maxima points

    WARNING: The code is not the final version of the task. So, it is
    very likely that it will be changed in the future.

    '''

    #Set up the X space of points
    X = np.ones(N*M)

    #Create the dataset
    for i in range(1, X.size):
        X[i] = X[i-1] + random.uniform(-1, 1)

    #Set up the minima and maxima vectors
    Ymin = np.zeros(X.size)
    Ymax = np.zeros(X.size)

    #Set up vector for combined representation of points (maxima and minima
    #at the smae time)
    Y = np.zeros(X.size)

    #Find local minima and maxima
    localMax = argrelextrema(X, np.greater)
    localMin = argrelextrema(X, np.less)

    #Combine maxima and minima in the vector Y
    for i in localMax:

        Y[i] = 2

    for i in localMin:

        Y[i] = 1

    #Previous value and index of the maxima or minima
    prevIndex = 0
    prevValue = 0

    print('Precompute mod dist')

    #Precompute the module of the distance in the space X
    #This could be implemented in more intelligent way to perform the task
    #in more uniform way
    collectionStd = [T * np.std(X[:-i] - X[i:]) for i in tqdm(range(1, 500))]

    print('Done')


    print('Choose the right Local Max - Min')

    #Iterate through the vector Y
    for i in tqdm(range(X.size)):

        #Continue if the point is not maxima or minima
        if Y[i] == 0:
            continue

        #If the points is not the first in the dataset and the current point
        #is not the same as the previous value (e.g., maxima and maxima)
        if prevIndex != 0 and Y[i] != prevValue:

            #Distance by index
            k = i - prevIndex

            #Distance should be more than 3 and not less than
            #D = T * np.std(X[:-k] - X[k:])
            if k > 3 and k >= collectionStd[k-1]:

                prevIndex = i
                prevValue = Y[i]

                #collect the maxima or minima point in Ymin or Ymax
                if Y[i] == 2:

                    Ymax[i] = 1

                else:

                    Ymin[i] = 1

        else:
            #If the point is the first in the dataset
            if prevValue == 0:

                prevIndex = i
                prevValue = Y[i]

                if Y[i] == 2:

                    Ymax[i] = 1

                else:

                    Ymin[i] = 1


    print('Done')

    return X, Ymin, Ymax
