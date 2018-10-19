#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 08:42:16 2018

@author: paul
"""

import numpy as np

import matplotlib.pyplot as plt
from tensorflow import keras as kr

def normalize(x):
    x_mean = np.mean(x)
    x_std = np.std(x)
    x_norm = (x-x_mean)/x_std
    return x_norm,x_mean,x_std

#%% =============================================================================
# Load trainings dataset
# =============================================================================
data_dir = "processed/"

#choose case 1 or 2
case = 1

# number of leading cars that are used for the labels
car = 0
n_leading_cars = 3  

# load pre processed data
Dx = np.loadtxt(data_dir+"case"+str(case)+"/headway.txt")
dotx = np.loadtxt(data_dir+"case"+str(case)+"/velocity.txt")
ddotx = np.loadtxt(data_dir+"case"+str(case)+"/acceleration.txt")
D_dotx = np.loadtxt(data_dir+"case"+str(case)+"/velocity_difference.txt")

# pre-arrays for features, filled with data to predict acceleration 
Dx_append = np.array(Dx[car:car+n_leading_cars,:])
dotx_append = np.array(dotx[car:car+n_leading_cars,:])
D_dotx_append = np.array(D_dotx[car:car+n_leading_cars,:])
#Dx_append = np.array(Dx[0:n_leading_cars,:])
#dotx_append = np.array(dotx[0:n_leading_cars,:])
#D_dotx_append = np.array(D_dotx[0:n_leading_cars,:])

# pre-array for labels, filled with acceleration
ddotx_append = np.array(ddotx[car,:])

# transpose array because time need to the first array dimension  
Dx    = Dx_append.T
dotx  = dotx_append.T
D_dotx  = D_dotx_append.T

# normalize input data (for a better ANN prediction)
Dx, meanDx, stdDx = normalize(Dx)
dotx, meandotx, stddotx = normalize(dotx)
D_dotx, meanD_dotx, stdD_dotx = normalize(D_dotx)

# merge features  into one big label array
#X = np.concatenate((Dx,dotx,D_dotx),axis=1)
X = np.concatenate((Dx,dotx),axis=1)

# rename label for a clearer code
y =  ddotx_append.T

#%% =============================================================================
# # build and train the model
# =============================================================================
model = kr.Sequential()
model.add(kr.layers.Dense(5, input_dim=len(X[0]),activation='relu'))
model.add(kr.layers.Dense(5, activation='relu'))
model.add(kr.layers.Dense(1, activation='linear'))

optimizer = kr.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=['mse'])

es = kr.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0.0,
                              patience=100,
                              verbose=0, mode='auto')
history = model.fit(X, y, epochs=500, batch_size=20,verbose=2,shuffle=True,validation_split=0.1,callbacks=[es])

model.save('model/model.h5')

np.savetxt("model/model_parameter.txt",[meanDx,stdDx,meandotx,stddotx,n_leading_cars])

#%% =============================================================================
# Plot history
# =============================================================================
plt.close("all")

# plot learning metric
plt.figure(1)

plt.plot(history.history['mean_squared_error'],label="MSE training")
plt.plot(history.history['val_mean_squared_error'],label="MSE validation")
plt.legend(frameon=False)

#%% show predicition against data

filename = 'model/model.h5'
model = kr.models.load_model(filename)

y_prediction = model.predict(X)
plt.figure(2)

plt.plot(y,label="data")
plt.plot(y_prediction,label="prediction")
plt.legend()

# plot weights of first layer
weights = model.layers[0].get_weights()
cm=plt.cm.bwr
C=plt.matshow(weights[0].T,cmap=cm)
plt.ylabel("to hidden layer neuron",fontsize=14)
plt.xlabel("from input neuron",fontsize=14)
plt.gca().xaxis.tick_bottom()
plt.colorbar(C).set_label(label="weights",size=14)