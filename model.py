#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 11:43:14 2018

@author: paul
"""
import numpy as np
from tensorflow import keras as kr
  
class model(object):
    def __init__(self,N,L,tmax,dt,xpert):            
        self.N =  N
        self.L =  L
        self.distance = np.arange(0,self.L,1) #array for distance
        self.xpert = xpert
        
        self.tmax  = tmax
        self.dt    = dt
        self.iters = abs(int(self.tmax/self.dt))
        self.time  = np.arange(0,self.tmax,self.dt)

        self.loaded_tf_model = kr.models.load_model('model/model.h5')
        self.meanDx,self.stdDx,self.meandotx,self.stddotx,self.n_leading_cars = np.loadtxt('model/model_parameter.txt')
        self.n_leading_cars  = int(self.n_leading_cars)
        
    def initCars(self,**kwargs):
        """
        initialise 0th time step
        """  
        
        self.x       = np.zeros(shape=(self.N,self.iters)) # position
        self.dot_x   = np.zeros(shape=(self.N,self.iters)) # velocity
        self.ddot_x  = np.zeros(shape=(self.N,self.iters)) # acceleration
        self.Delta_x = np.zeros(shape=(self.N,self.iters)) # headway
        
        self.x[:,0]      = np.linspace(0,self.L,self.N) 
        self.dot_x[:,0]  = self.meandotx         
        self.ddot_x[:,0] = 0.
        
        self.x[:,0] = self.x[:,0] + self.xpert
        self.Delta_x[:,0]   = self.headway(self.x[:,0],self.L)    
        
    def integrate(self,**kwargs):
        """
        Integrate the model using a fortran or a python kernel 
        """
        for i in range(0,self.iters-1):
            self.integration_procedure(i)
            
    def integration_procedure(self,i):
        """
        RK4 integration scheme
        """
        h = self.dt
        k1 = self.acceleration(self.Delta_x[:,i],self.dot_x[:,i])
        self.dot_x[:,i+1] = self.dot_x[:,i] + k1*h/2
        
        k2 = self.acceleration(self.Delta_x[:,i],self.dot_x[:,i+1])
        
        self.dot_x[:,i+1] = self.dot_x[:,i] + k2*h/2
        k3 = self.acceleration(self.Delta_x[:,i],self.dot_x[:,i+1])
        
        self.dot_x[:,i+1] = self.dot_x[:,i] + k3*h
        k4 = self.acceleration(self.Delta_x[:,i],self.dot_x[:,i+1])
        
        self.ddot_x[:,i+1] = k1 
        
        self.dot_x[:,i+1] = self.dot_x[:,i] + h/6. * (k1 + 2*k2 + 2*k3 + k4) 
        
        # just allow postive velocities
        self.dot_x[self.dot_x[:,i+1]<0,i+1] = 0.
        
        self.x[:,i+1]      = self.x[:,i] + self.dot_x[:,i+1] * h
        
        self.x[:,i+1]      = self.x[:,i+1]%self.L

        # Diagnostics
        self.Delta_x[:,i+1]   = self.headway(self.x[:,i+1],self.L)

    def acceleration(self,Delta_x,dot_x):
        """
        returns the acceleration of cars based on an ANN
        """
        
        Dx = Delta_x.reshape((len(Delta_x),1))
        dotx = dot_x.reshape((len(Delta_x),1))
                
        Dx = (Dx - self.meanDx)/self.stdDx
        dotx = (dotx - self.meandotx)/self.stddotx
        
        X = np.concatenate((Dx[0:self.n_leading_cars,:].T,dotx[0:self.n_leading_cars,:].T),axis=1)

        X_all = np.zeros((self.N,len(X[0])))
        
        # for car number 0
        X_all[0,:] = X
        
        for i in range(self.N-1):    
            Dx = np.roll(Dx,-1,axis=0)
            dotx = np.roll(dotx,-1,axis=0)            
            X_all[i+1,:] =  np.concatenate((Dx[0:self.n_leading_cars,:].T,dotx[0:self.n_leading_cars,:].T),axis=1)
            
        ddotx = self.loaded_tf_model.predict(X_all)
        
        return ddotx[:,0]
    
    def headway(self,x,L):
        Dx = np.zeros(self.N)
        Dx[:-1] = ((x[1:] - x[:-1])+L)%L
        Dx[-1] = ((x[0] - x[-1])+L)%L
        return Dx 
    
    