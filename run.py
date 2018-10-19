#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 16:09:01 2018

@author: paul
"""

from model import model
from plot import plots

import numpy as np
import matplotlib.pyplot as plt
import time

print "Run ANN model #################"

N = 22
L = 230
dt =  0.1
tmax = 100
xpert= np.zeros(N)
xpert = 15*np.sin(2*np.pi/float(N)*np.arange(N))

# Model simulation
start = time.time() 

m = model(N,L,tmax,dt,xpert)
m.initCars()
m.integrate() 

end = time.time()

print "calculation time: "+str(end-start)

# plot various variables
plt.close("all")

plots(m)