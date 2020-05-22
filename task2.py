import random
import numpy as np
import matplotlib.pyplot as plt
import task1

def plotGeneratedData(X, Ymin, Ymax, N):

    '''
    Task 2: Plot randomly picked part of the generated dataset

    The X as a line
    Ymin - blue dots
    Ymax - red dots

    The script saves the figure as a png image.
    The example could be seen in the repo.

    '''

    #Randomly choose the range
    i = random.randint(0, X.size - N)

    plt.figure(figsize=(20,10))

    plotRange = X[i:i+N]

    plotYmin = Ymin[i:i+N]
    plotYmax = Ymax[i:i+N]

    #Plot the range of x as a line
    plt.plot(range(plotRange.size), plotRange, linewidth = 1)

    #Plot Ymin and Ymax
    for i in range(plotYmin.size):

        if Ymin[i] == 1:

            plt.scatter(i, plotRange[i], c = 'r', s = 100)

        if Ymax[i] == 1:

            plt.scatter(i, plotRange[i], c = 'b', s = 100)

    #Save as image
    plt.savefig('ExamplePlotTask2.png')
    plt.close()



X, Ymin, Ymax = task1.generateDataset()
plotGeneratedData(X, Ymin, Ymax, 10000)
