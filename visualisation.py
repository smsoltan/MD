"""
Modified examples from matplotlib.org and stackexchange
"""
import numpy as np
import matplotlib.pyplot as plt

import sys


name = input("enter the name of the datafile produced by MD program: ")
f = open(name, "r")
ignore = f.readline()
raw_data = f.readlines()
f.close()

data = np.array([[float(i) for i in s.split()] for s in raw_data])
data = np.transpose(data)



fig, axs = plt.subplots(2, 1)
axs[0].plot(data[0], data[1])
axs[0].plot(data[0], data[3])
#axs[0].set_xlim(0, 2)
axs[0].set_xlabel('time')
axs[0].set_ylabel('kinetic and total')
axs[0].grid(True)

axs[1].plot(data[0], data[2])
#axs[1].plot(data[0], data[3])
axs[1].set_ylabel('potential')
axs[1].grid(True)


fig.tight_layout()

name = input("enter where to save png: ")
fig.savefig(name)
