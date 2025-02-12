#!/usr/bin/env python
# python plotting script

from matplotlib import pyplot as plt
import numpy as np
import sys

# set up the figure
fig, axes = plt.subplots(1, len(sys.argv[1:]))

ind = 0
for filename in sys.argv[1:]:

    # Load data from text file (x, y, z)
    data = np.loadtxt(filename)
    print(data)

    # Sort the data
    data = data[data[:,0].argsort()]
    x = data[:,1]
    y = data[:,0]
    z = data[:,2]
    x = np.unique(x)
    x = np.sort(x)
    y = np.unique(y)
    y = np.sort(y)[::-1]

    # Create a 2D grid of data
    Z = np.zeros((len(y), len(x)))
    for dataPoint in data:
        Z[np.where(y == dataPoint[0])[0][0], np.where(x == dataPoint[1])[0][0]] = dataPoint[2]

    # Plot the data as a subfigure
    im = axes[ind].imshow(Z, cmap='jet', interpolation='none', aspect='auto', extent=[x[0], x[-1], y[-1], y[0]])
    axes[ind].set_xlabel('J_perp')
    axes[ind].set_ylabel('J_x')
    axes[ind].set_title(filename)
    ind += 1

plt.tight_layout()
plt.show()




