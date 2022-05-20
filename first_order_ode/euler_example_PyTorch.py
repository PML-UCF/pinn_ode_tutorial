import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as matplotlib

from torch import nn, Tensor, sqrt, pi, stack, optim


class Normalization(nn.Module):
    def __init__(self, S_low, S_up, a_low, a_up, **kwargs):
        super(Normalization, self).__init__(**kwargs)
        self.low_bound_S   = S_low
        self.upper_bound_S = S_up
        self.low_bound_a   = a_low
        self.upper_bound_a = a_up

    def forward(self, x):
        
        s = x[:,0]
        a = x[:,1]
        
        s  = (s - self.low_bound_S) / (self.upper_bound_S - self.low_bound_S)
        a  = (a - self.low_bound_a) / (self.upper_bound_a - self.low_bound_a)
        
        return stack((s,a)).T


class EulerIntegrator(nn.Module):
    def __init__(self, C, m, F, **kwargs):
        super(EulerIntegrator, self).__init__(**kwargs)
        self.C = C
        self.m = m
        self.F = F
    
    def forward(self, x, a0):
        
        # x is a tensor of shape (sequence_length, input_size)
        (sequence_length, input_size) = x.shape
        
        a_t  = a0
        a    = a0.repeat(sequence_length,1)
        for t in range(sequence_length):
            x_t  = x[t, :]
            dk_t = self.F * x_t * sqrt(pi * a_t)
            da_t = self.C * (dk_t ** self.m)
            a_t  = a_t + da_t
            a[t] = a_t
        
        return a


class HybridEulerIntegrator(nn.Module):
    def __init__(self, C, m, dKlayer, **kwargs):
        super(HybridEulerIntegrator, self).__init__(**kwargs)
        self.C = C
        self.m = m
        self.dKlayer = dKlayer
    
    def forward(self, x, a0):
        
        # x is a tensor of shape (sequence_length, input_size)
        (sequence_length, input_size) = x.shape
        
        a_t  = a0
        a    = a0.repeat(sequence_length,1)
        for t in range(sequence_length):
            x_t  = x[t, :]
            dk_t = self.dKlayer(stack((x_t, a_t),1))[:,0]
            da_t = self.C * (dk_t ** self.m)
            a_t  = a_t + da_t
            a[t] = a_t
        
        return a
        
        
if __name__ == "__main__":
    # Paris law coefficients
    [C, m] = [1.5E-11, 3.8]
    F = 1.0
    
    # data
    Strain = np.asarray(pd.read_csv('./data/Strain.csv')).T
    Strain = Tensor(Strain)
    
    atrain = np.asarray(pd.read_csv('./data/atrain.csv')).T
    atrain = Tensor(atrain)
    
    a0 = np.asarray(pd.read_csv('./data/a0.csv'))[0,0]
    a0 = Tensor(a0*np.ones((Strain.shape[1])))
    
    PINN  = EulerIntegrator(C=C, m=m, F=F)
    aPINN = PINN(Strain, a0)
    
    "-------------------------------------------------------------------------"
    " dKlayer and weight initialization "
    dKlayer = nn.Sequential(Normalization(Strain.min(), Strain.max(), atrain.min(), atrain.max()),
                            nn.Linear(2, 5),
                            nn.Tanh(),
                            nn.Linear(5, 1))

    S_range  = np.linspace(Strain.min(), Strain.max(), 1000)
    a_range  = np.linspace(atrain.min(), atrain.max(), 1000)[np.random.permutation(np.arange(1000))]
    dK_range = Tensor(np.array(-12.05 + 0.24 * S_range + 760.0 * a_range)[:,np.newaxis])
    inputs_train = Tensor(np.transpose(np.asarray([S_range, a_range])))

    loss_fn = nn.MSELoss()
    learning_rate = 1e-2
    optimizer = optim.RMSprop(dKlayer.parameters(), lr=learning_rate)
    
    epochs = 2000
    dKlayer.eval()
    dKlayer.train()
    for t in range(epochs):
        optimizer.zero_grad()
        Ypred = dKlayer(inputs_train)
        loss = loss_fn(Ypred, dK_range)
        loss.backward()
        optimizer.step()
        
        print(t, loss.item())
            
    "-------------------------------------------------------------------------"
    " hPINN training "
    hPINN  = HybridEulerIntegrator(C=C, m=m, dKlayer=dKlayer)
    aPred_before = hPINN(Strain, a0)

    loss_fn = nn.MSELoss()
    learning_rate = 1e-2
    optimizer = optim.RMSprop(hPINN.parameters(), lr=learning_rate)
    
    epochs = 100
    hPINN.eval()
    hPINN.train()
    for t in range(epochs):
        optimizer.zero_grad()
        y_pred = hPINN(Strain, a0)
        loss = loss_fn(y_pred[-1,:], atrain[0,:])
        loss.backward()
        optimizer.step()
        print(t, loss.item())


    aPred = hPINN(Strain, a0)
    
    "-------------------------------------------------------------------------"
    " plottting "
    matplotlib.rc('font', size=12)
    ifig = 0
    
    ifig = ifig + 1
    fig  = plt.figure(ifig)
    fig.clf()

    plt.plot([0,0.05],[0,0.05],'--k')
    plt.plot(atrain[0,:], aPred_before.detach()[-1,:], 'o', label = 'before training')
    plt.plot(atrain[0,:], aPred.detach()[-1,:], 's', label = 'after training')
    plt.xlabel("actual crack length (m)")
    plt.ylabel("predicted crack length (m)")
    plt.title("PyTorch")
    plt.legend(loc = 'upper center',facecolor = 'w')
    plt.grid(which='both')
    plt.show()
    
