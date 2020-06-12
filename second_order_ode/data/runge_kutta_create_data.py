import numpy as np
from scipy import signal
from scipy import linalg
import pandas as pd

import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    #--------------------------------------------------------------------------
    m1 = 20.0
    m2 = 10.0
    
    k1 = 2e3
    k2 = 1e3
    k3 = 5e3
    
    c1 = 100.0
    c2 = 110.0
    c3 = 120.0
    
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
    
    yTarget = yScipy + 1.5e-5*np.random.randn(yScipy.shape[0], yScipy.shape[1])
    
    df = pd.DataFrame(np.hstack([t[:,np.newaxis],u,yScipy,yTarget]), columns=['t', 'u0','u1','y0','y1','yT0','yT1'])
    
    df.to_csv('./data.csv', index = False)
    
    #--------------------------------------------------------------------------  
    plt.plot(t, yTarget, '-', color ='gray')
    plt.plot(t, yScipy, '-', color ='r')
    plt.xlabel('time')
    plt.ylabel('displacement')
    plt.grid('on')
    plt.show()
