# ______          _           _     _ _ _     _   _      
# | ___ \        | |         | |   (_) (_)   | | (_)     
# | |_/ / __ ___ | |__   __ _| |__  _| |_ ___| |_ _  ___ 
# |  __/ '__/ _ \| '_ \ / _` | '_ \| | | / __| __| |/ __|
# | |  | | | (_) | |_) | (_| | |_) | | | \__ \ |_| | (__ 
# \_|  |_|  \___/|_.__/ \__,_|_.__/|_|_|_|___/\__|_|\___|
# ___  ___          _                 _                  
# |  \/  |         | |               (_)                 
# | .  . | ___  ___| |__   __ _ _ __  _  ___ ___         
# | |\/| |/ _ \/ __| '_ \ / _` | '_ \| |/ __/ __|        
# | |  | |  __/ (__| | | | (_| | | | | | (__\__ \        
# \_|  |_/\___|\___|_| |_|\__,_|_| |_|_|\___|___/        
#  _           _                     _                   
# | |         | |                   | |                  
# | |     __ _| |__   ___  _ __ __ _| |_ ___  _ __ _   _ 
# | |    / _` | '_ \ / _ \| '__/ _` | __/ _ \| '__| | | |
# | |___| (_| | |_) | (_) | | | (_| | || (_) | |  | |_| |
# \_____/\__,_|_.__/ \___/|_|  \__,_|\__\___/|_|   \__, |
#                                                   __/ |
#                                                  |___/ 
#														  
# MIT License
# 
# Copyright (c) 2019 Probabilistic Mechanics Laboratory
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

""" Train physics-informed recursive neural network
"""

import numpy as np
from scipy import signal
from scipy import linalg
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as matplotlib

if __name__ == "__main__":
    
    #--------------------------------------------------------------------------
#    m1 = 20.0
#    m2 = 10.0
#    
#    k1 = 2e3
#    k2 = 1e3
#    k3 = 5e3
#    
#    c1 = 100.0
#    c2 = 110.0
#    c3 = 120.0
    
    m1 = 15.0
    m2 = 20.0
    
    k1 = 1.0e3
    k2 = 1.5e3
    k3 = 2.0e3
    
    c1 = 60.0
    c2 = 70.0
    c3 = 80.0
    
    Mvib = np.asarray([[m1, 0.0], [0.0, m2]], dtype = float)
    Cvib = np.asarray([[c1+c2, -c2], [-c2, c2+c3]], dtype = float) 
    Kvib = np.asarray([[k1+k2, -k2], [-k2, k2+k3]], dtype = float)

    #--------------------------------------------------------------------------
    # building matrices in continuous time domain
    n = Mvib.shape[0]
    I = np.eye(n)
    Z = np.zeros([n,n])
    Minv = linalg.pinv(Mvib)
    
    negMinvK = - np.matmul(Minv, Kvib)
    negMinvC = - np.matmul(Minv, Cvib)
    
    Ac = np.hstack((np.vstack((Z,negMinvK)), np.vstack((I,negMinvC))))
    Bc = np.vstack((Z,Minv))
    Cc = np.hstack((I,Z))
    Dc = Z.copy()
    
    systemC = (Ac, Bc, Cc, Dc)
    
    #--------------------------------------------------------------------------
    # building matrices in discrete time domain
    t = np.linspace(0,2,1001,dtype = float)
    dt = t[1] - t[0]
    
    sD = signal.cont2discrete(systemC, dt)
    
    Ad = sD[0]
    Bd = sD[1]
    Cd = sD[2]
    Dd = sD[3]
    
    systemD = (Ad, Bd, Cd, Dd, dt)
    
    #--------------------------------------------------------------------------
    u = np.zeros((t.shape[0], n))
    u[:, 0] = np.ones((t.shape[0],))
    
    x0 = np.zeros((Ad.shape[1],), dtype = 'float32')
    
    output = signal.dlsim(systemD, u = u, t = t, x0 = x0)
    yScipy = output[1]
    
    yTarget = yScipy + 5e-6*np.random.randn(yScipy.shape[0], yScipy.shape[1])
    
    df = pd.DataFrame(np.hstack([t[:,np.newaxis],u,yScipy,yTarget]), columns=['t', 'u0','u1','y0','y1','yT0','yT1'])
    
    df.to_csv('./data02.csv', index = False)
    
    #--------------------------------------------------------------------------
    matplotlib.rc('font', size=14)

    ifig = 1
    fig = plt.figure(ifig)
    fig.clf()
    
    plt.plot(t, yTarget, '-', color ='gray')
    plt.plot(t, yScipy, '-', color ='r')
    plt.xlabel('time')
    plt.ylabel('displacement')
    plt.grid('on')
    plt.show()
