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

# =============================================================================
# Load trainings dataset
# =============================================================================
data_dir = "processed/"

#choose case 1 or 2
case = 1

# number of leading cars that are used for the labels
car = 0
n_leading_cars = 2

RNN = True

# load pre processed data
Dx = np.loadtxt(data_dir+"case"+str(case)+"/headway.txt")
dotx = np.loadtxt(data_dir+"case"+str(case)+"/velocity.txt")
ddotx = np.loadtxt(data_dir+"case"+str(case)+"/acceleration.txt")
D_dotx = np.loadtxt(data_dir+"case"+str(case)+"/velocity_difference.txt")

# pre-arrays for features, filled with data to predict acceleration 
Dx_append = np.array(Dx[car:car+n_leading_cars,:])
dotx_append = np.array(dotx[car:car+n_leading_cars,:])
D_dotx_append = np.array(D_dotx[car:car+n_leading_cars,:])
# pre-array for labels, filled with acceleration
ddotx_append = np.array(ddotx[car,:])

#for i in range(1,3):
#    Dx_append = np.concatenate((Dx_append,np.roll(Dx,i,axis=1)[0:n_leading_cars,:]),axis=0)
#    dotx_append = np.concatenate((dotx_append,np.roll(dotx,i,axis=1)[0:n_leading_cars,:]),axis=0)

### use all data
#for i in range(1,5):
#    Dx_append = np.concatenate((Dx_append,np.roll(Dx,i,axis=0)[0:n_leading_cars,:]),axis=1)
#    dotx_append = np.concatenate((dotx_append,np.roll(dotx,i,axis=0)[0:n_leading_cars,:]),axis=1)
#    D_dotx_append = np.concatenate((D_dotx_append,np.roll(D_dotx,i,axis=0)[0:n_leading_cars,:]),axis=1)
#    ddotx_append = np.concatenate((ddotx_append,np.roll(ddotx,i,axis=0)[0,:]),axis=0)

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

if RNN:
    X = X.reshape(X.shape[0],1,X.shape[1])
    #Xnew = np.concatenate((X,np.roll(X,1,axis=0)),axis=1)
    #X = Xnew
    
# rename label for a clearer code
y =  ddotx_append.T

#%% =============================================================================
# # build and train the model
# =============================================================================
model = kr.Sequential()

if RNN:
    model.add(kr.layers.LSTM(100, activation='relu', input_shape=(X.shape[1],X.shape[2])))
    model.add(kr.layers.Dropout(0.2))
    model.add(kr.layers.Dense(10, activation='relu'))

else:
    model.add(kr.layers.Dense(100, input_dim=len(X[0]),activation='relu'))
    model.add(kr.layers.Dropout(0.2))
    model.add(kr.layers.Dense(50, activation='relu'))
    model.add(kr.layers.Dropout(0.2))
    model.add(kr.layers.Dense(10, activation='relu'))
    
model.add(kr.layers.Dense(1, activation='linear'))

optimizer = kr.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., amsgrad=False)

model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=['mse'])

es = kr.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0.0,
                              patience=500,
                              verbose=0, mode='auto')

history = model.fit(X, y, epochs=10000, batch_size=X.shape[0],verbose=2,shuffle=True,validation_split=0.1,callbacks=[es])

model.save('model/model.h5')

np.savetxt("model/model_parameter.txt",[meanDx,stdDx,meandotx,stddotx,n_leading_cars,RNN])

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
C=plt.matshow(weights[0].T,cmap=cm,vmin=-1,vmax=1)
plt.ylabel("to hidden layer neuron",fontsize=14)
plt.xlabel("from input neuron",fontsize=14)
plt.gca().xaxis.tick_bottom()
plt.colorbar(C).set_label(label="weights",size=14)

C2=plt.matshow(weights[1].T,cmap=cm,vmin=-1,vmax=1)
plt.ylabel("to hidden layer neuron",fontsize=14)
plt.xlabel("from input neuron",fontsize=14)
plt.gca().xaxis.tick_bottom()
plt.colorbar(C2).set_label(label="weights",size=14)