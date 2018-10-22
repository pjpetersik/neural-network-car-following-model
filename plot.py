#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 18:25:54 2018

@author: paul
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import griddata

fs = 14
    
def plots(model):
    
# =============================================================================
#   hystersis loop
# =============================================================================
    fig, ax = plt.subplots()
    time = model.time
    x = model.x
    dot_x = model.dot_x
    Delta_x = model.Delta_x
    distance = model.distance
    
    car =  0
    start = 0
    end   = model.iters
    iters = end - start
    jump = 1  # just plot every 3rd iteration to save time
    
    fs =14
    c = np.linspace(model.time[start],model.time[end-1],iters)
    
    #ax.set_title("velocity vs. headway, car=" + str(car))
    ax_scatter = ax.scatter(Delta_x[car,start:end:jump],dot_x[car,start:end:jump],marker="x",s=10,c=c[::jump])
    ax.set_xlabel('headway [m]', fontsize = fs)
    ax.set_ylabel('velocity [s]',fontsize = fs)
    #ax.set_ylim(0,10)
    #ax.set_xlim(0,15)
    ax.tick_params(direction="in")
    ax.set_title(r'ANN, $L=$'+str(model.L))
    cb=fig.colorbar(ax_scatter, ax=ax)
    cb.set_label(label="time [s]",size=fs)
# =============================================================================
#     headway velocities
# =============================================================================
    fig, ax = plt.subplots()
    for j in np.arange(0,model.N,jump):
        diffx = np.roll(x[j,:],-1)-x[j,:]
        masked_x = np.ma.array(x[j,:])
        masked_x[diffx<-200] = np.ma.masked 
        ax.plot(time,masked_x,lw=0.8,c="red")
    #ax.set_title("car positions", fontsize = fs)
    ax.set_ylabel("position [m]", fontsize = fs)
    ax.set_xlabel("time [s]", fontsize = fs)
    ax.tick_params(direction="in")
# =============================================================================
#     hmodelöller velocites
# =============================================================================
    fig, ax = plt.subplots()
    jump = int(model.tmax/100)  # just consider every 100 iteration for the interpolation to save time
    x_data = x[:,::jump]
    dot_x_data = dot_x[:,::jump]
    t_data = time[::jump]
    lent = len(t_data)
    
    grid_x, grid_t = np.meshgrid(distance,time)
    x_point =  x_data.reshape(model.N*lent,1)
    t_point =  np.tile(t_data,model.N)
    t_point =  t_point.reshape(model.N*lent,1)
    points = np.concatenate((x_point,t_point),axis=1)
    dot_x_values = dot_x_data.reshape(model.N*lent)
    grid_dot_x = griddata(points, dot_x_values, (grid_x, grid_t), method='linear')
    
    cmap = "inferno"
    contours = np.linspace(0.,9,21)
    cf = ax.contourf(time,distance,grid_dot_x.T,contours,cmap=cmap, extend="both")
    ax.set_xlabel("time [s]", fontsize = fs)
    ax.set_ylabel("position [m]", fontsize = fs)
    ax.tick_params(direction="in")
    #ax.set_title("velocity", fontsize = fs)
    cb=fig.colorbar(cf, ax=ax)
    cb.set_label(label="velocity [m/s]", size=14)
# =============================================================================
# standard deviation headway
# =============================================================================
    fig, ax = plt.subplots()
    #ax.set_title("std($\Delta$x) vs. t")
    ax.plot(time,Delta_x.std(axis=0))
    ax.set_xlabel("time [s]",fontsize = fs)
    ax.set_ylabel("std($\Delta$x) [m]",fontsize = fs)
    ax.tick_params(direction="in")
    #ax.set_xlim(0,250)
    #ax.set_ylim(0,6)
    
# =============================================================================
# standard deviation velocity
# =============================================================================
    fig, ax = plt.subplots()
    #ax.set_title("std($\Delta$x) vs. t")
    ax.plot(time,dot_x.std(axis=0))
    ax.set_xlabel("time [s]",fontsize = fs)
    ax.set_ylabel("std($\dot{x}$) [m/s]",fontsize = fs)
    ax.tick_params(direction="in")
    #ax.set_xlim(0,250)
    #ax.set_ylim(0,4)