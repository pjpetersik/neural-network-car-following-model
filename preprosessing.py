#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 18:39:50 2018

@author: paul
"""

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
plt.close("all")
from scipy.interpolate import griddata


#%% =============================================================================
# Load trainings dataset
# =============================================================================

# cirucuit length
L = 230

# data directiry
input_directory = "raw/"
output_directory = "processed/"

# read data
case = 2
raw_data = pd.read_csv(input_directory+"case"+str(case)+".data",skiprows=14,sep='\s+', names=["position","time"],skip_blank_lines=False)

raw_position = raw_data["position"]
raw_time     = raw_data["time"]

# get array sizes 
n_cars = len(raw_time.index[raw_time.apply(np.isnan)])  # count NaNs (they devide measurements of different cars)
time_steps = raw_time.index[raw_time.apply(np.isnan)][0] # index value first NaN is the number of time steps

# drop the NaNs 
raw_position = raw_position.dropna()
raw_time     = raw_time.dropna()

# reshape the array and take account that data have a NaN value is "missing" for case 1
try:
    position = np.array(raw_position).reshape((n_cars,time_steps))
    time = np.array(raw_time).reshape((n_cars,time_steps))
    
except:
    n_cars = n_cars+1
    position = np.array(raw_position).reshape((n_cars,time_steps))
    time = np.array(raw_time).reshape((n_cars,time_steps))


#%% =============================================================================
# pre process data 
# =============================================================================

# arrays to store velocity, acceleration, headway and velocity difference     
velocity = np.zeros_like(position)
acceleration = np.zeros_like(position)
headway = np.zeros_like(position)
D_velocity = np.zeros_like(position)

# use central differences in space and time 
dx = (position[:,2:] - position[:,:-2])
dt = (time[:,2:] - time[:,:-2])

# use the modular function to account for the fact that position data was saved 
dx[dx<-200] = dx[dx<-200]%L

# compute the variables
velocity[:,1:-1] = dx/dt
acceleration[:,2:-2] = (velocity[:,3:-1] - velocity[:,1:-3])/(time[:,3:-1] - time[:,1:-3])
headway[:,:] = (np.roll(position,-1,axis=0)-position[:,:])%L
D_velocity[:,:] = (np.roll(velocity,-1,axis=0)-velocity[:,:])%L  

# drop values at the beginning and the end (they are 0 for acceleration)
t = time[:,2:-2]
x = position[:,2:-2]
Dx = headway[:,2:-2]
dotx = velocity[:,2:-2]
D_dotx = D_velocity[:,2:-2]
ddotx = acceleration[:,2:-2]

# shift target (prediction of the acceleration of the next time step i+1)
ddotx = np.roll(ddotx,-1,axis=1)

#delete last array entry since it loses meaning due to shifting
t= t[:,:-1]
x = x[:,:-1]
Dx = Dx[:,:-1]
dotx = dotx[:,:-1]
D_dotx = D_dotx[:,:-1]
ddotx = ddotx[:,:-1]

#save data 
np.savetxt(output_directory+"case"+str(case)+"/position.txt",x)
np.savetxt(output_directory+"case"+str(case)+"/headway.txt",Dx)
np.savetxt(output_directory+"case"+str(case)+"/velocity.txt",dotx)
np.savetxt(output_directory+"case"+str(case)+"/acceleration.txt",ddotx)
np.savetxt(output_directory+"case"+str(case)+"/velocity_difference.txt",D_dotx)
np.savetxt(output_directory+"case"+str(case)+"/time.txt",t)

#%% plot 

fig, ax = plt.subplots()


car =  0
start = 0
end   = len(Dx[0,:])
iters = end - start
jump = 1  # just plot every 3rd iteration to save time

fs =14
c = np.linspace(0,np.max(time),end)

#ax.set_title("velocity vs. headway, car=" + str(car))
ax_scatter = ax.scatter(Dx[car,:],dotx[car,:], marker="x",s=10,c=c)
ax.set_xlabel(r'$\Delta x$', fontsize = fs)
ax.set_ylabel(r'$\dot{x}$',fontsize = fs)
ax.set_ylim(-1,12)
ax.set_xlim(0,35)
ax.tick_params(direction="in")
cb=fig.colorbar(ax_scatter, ax=ax)
cb.set_label(label="time [s]",size=fs)

#%% =============================================================================
#     headway velocities
# =============================================================================
fig, ax = plt.subplots()

for j in np.arange(0,n_cars):
    diffx = np.roll(x[j,:],-1)-x[j,:]
    masked_x = np.ma.array(x[j,:])
    masked_x[diffx<-200] = np.ma.masked 
    ax.plot(t[j,:],masked_x,lw=0.8,c="red")

#ax.set_title("car positions", fontsize = fs)
ax.set_ylabel("position [m]", fontsize = fs)
ax.set_xlabel("time [s]", fontsize = fs)
ax.set_ylim(0,230)
ax.set_xlim(0,np.max(time))
ax.tick_params(direction="in")

#%% =============================================================================
# velocities hovmöller
# =============================================================================
fig, ax = plt.subplots()

distance_arr=np.linspace(0,230,100)
time_arr = np.linspace(0,np.max(time),100)

grid_x, grid_t = np.meshgrid(distance_arr,time_arr)

x_point =  x.reshape(x.size,1)
t_point =  t.reshape(t.size,1)
points = np.concatenate((x_point,t_point),axis=1)
dot_x_values = dotx.reshape(dotx.size,1)

grid_dot_x = griddata(points, dot_x_values[:,0], (grid_x, grid_t), method='linear')


cmap = "inferno"
contours = np.linspace(0,9,21)
cf = ax.contourf(time_arr,distance_arr,grid_dot_x.T,contours,cmap=cmap, extend="max")
ax.set_xlabel("time [s]", fontsize = fs)
ax.set_ylabel("position [m]", fontsize = fs)
ax.tick_params(direction="in")
cb=fig.colorbar(cf, ax=ax)
cb.set_label(label="velocity [m/s]", size=14)


#%% =============================================================================
# standard deviation headway
# =============================================================================
fig, ax = plt.subplots()
#ax.set_title("std($\Delta$x) vs. t")
ax.plot(t[0,:],Dx.std(axis=0))
ax.set_xlabel("time [s]",fontsize = fs)
ax.set_ylabel("std($\Delta$x) [m]",fontsize = fs)
ax.tick_params(direction="in")

ax.set_xlim(0,250)
ax.set_ylim(0,6)



#%% =============================================================================
# standard deviation velocity differenc
# =============================================================================
fig, ax = plt.subplots()
#ax.set_title("std($\Delta$x) vs. t")
ax.plot(t[0,:],dotx.std(axis=0))
ax.set_xlabel("time [s]",fontsize = fs)
ax.set_ylabel("std($\dot{x}$) [m/s]",fontsize = fs)
ax.tick_params(direction="in")

ax.set_xlim(0,250)
ax.set_ylim(0,4)
