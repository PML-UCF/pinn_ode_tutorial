#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:28:18 2020

@author: felipeviana
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as matplotlib

def myfun(invM,K,C,u,x,xdot):
    out = np.matmul(invM, u - np.matmul(C,xdot) - np.matmul(K, x))
    return out

if __name__ == "__main__":
    # mass, spring coefficient, damping coefficient
    m = np.array([20.0, 10.0])
    c = np.array([10.0, 10.0, 10.0])
    k = np.array([2e3, 1e3, 5e3])
    
    M = np.array([[m[0], 0.], [0., m[1]]])
    C = np.array([[c[0]+c[1], -c[1]], [-c[1], c[1]+c[2]]])
    K = np.array([[k[0]+k[1], -k[1]], [-k[1], k[1]+k[2]]])
    invM = np.linalg.inv(M)

    # data
    df = pd.read_csv('data.csv')
    t  = df[['t']].values
    dt = (t[1] - t[0])[0]
    u = df[['u0', 'u1']].values
    yxp = df[['yT0', 'yT1']].values

    z0 = np.zeros((2*len(m),)) # z = [y ydot]
    
    # Runge-Kutta integration
    A = np.array([0., 0.5, 0.5, 0.1])
    B = np.array([1/6, 2/6, 2/6, 1/6])
    
    N = len(t)
    y    = np.zeros_like(u)
    ydot = np.zeros_like(u)
    y[0,:]    = z0[0:2]
    ydot[0,:] = z0[3:4]
    for i in range(N-1):
        ui = u[i,:]
        yi = np.repeat(np.array([y[i,:]]),4,axis=0)
        ydoti  = np.repeat(np.array([ydot[i,:]]),4,axis=0)
        yddoti = myfun(invM,K,C,ui,yi[0,:],ydoti[0,:])
        fn = np.zeros([4,yddoti.shape[0]])
        for j in range(4):
            yi[j,:]    = yi[j,:] + A[j]*ydoti[j,:]*dt
            ydoti[j,:] = ydoti[j,:] + A[j]*yddoti*dt
            fn[j,:]    = myfun(invM,K,C,ui,yi[j,:],ydoti[j,:])
            
        y[i+1,:]    = y[i,:]    + np.matmul(B,ydoti)*dt
        ydot[i+1,:] = ydot[i,:] + np.matmul(B,fn)*dt
        
    plt.plot(t, y)
    plt.plot(t, yxp, 'gray')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.grid('on')
    plt.show()
    