import random
import numpy as np
import matplotlib.pyplot as plt
import task1

def plotGeneratedData(X, Ymin, Ymax, N):

    i = random.randint(0, X.size - N)

    plt.figure(figsize=(20,10))

    plotRange = X[i:i+N]

    plotYmin = Ymin[i:i+N]
    plotYmax = Ymax[i:i+N]

    plt.plot(range(plotRange.size), plotRange, linewidth = 1)

    for i in range(plotYmin.size):

        if Ymin[i] == 1:

            plt.scatter(i, plotRange[i], c = 'r', s = 100)

        if Ymax[i] == 1:

            plt.scatter(i, plotRange[i], c = 'b', s = 100)

    plt.savefig('ExamplePlotTask2.png')
    plt.close()



X, Ymin, Ymax = task1.generateDataset()
plotGeneratedData(X, Ymin, Ymax, 10000)
