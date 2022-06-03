import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as matplotlib

from torch import (
    nn,
    Tensor,
    sqrt,
    pi,
    stack,
    cat,
    randperm,
    min, 
    max,
    linspace,
    transpose, 
    optim
    )


class Normalization(nn.Module):
    def __init__(self, S_low, S_up, a_low, a_up, **kwargs):
        super(Normalization, self).__init__(**kwargs)
        self.low_bound_S = S_low
        self.upper_bound_S = S_up
        self.low_bound_a = a_low
        self.upper_bound_a = a_up

    def forward(self, inputs):

        output_S = (inputs[:, 0] - self.low_bound_S)/(self.upper_bound_S - self.low_bound_S)
        output_a = (inputs[:, 1] - self.low_bound_a)/(self.upper_bound_a - self.low_bound_a)

        return stack([output_S,output_a], dim=1)


class MyRNN(nn.Module):
    def __init__(self, cell):
        super(MyRNN, self).__init__()
        self.cell = cell

    def forward(self, inputs, a0):

        bs, seq_sz, _ = inputs.shape
        a = a0
        for t in range(seq_sz): 
            input = inputs[:, t, :]
            state = self.cell.forward(input, a) 
            a = a+state

        return a


class EulerIntegratorCell(nn.Module):
    def __init__(self, C, m, dKlayer, **kwargs):
        super(EulerIntegratorCell, self).__init__(**kwargs)
        self.C = C
        self.m = m
        self.dKlayer = dKlayer

    def forward(self, inputs, states): 

        a_tm1 = states
        x_d_tm1 = cat((inputs, a_tm1), dim=1)
        dk_t = self.dKlayer(x_d_tm1)
        da_t = self.C * (dk_t ** self.m)

        return da_t


class DkPhys(nn.Module):
    def __init__(self, F, **kwargs):
        super(DkPhys, self).__init__(**kwargs)
        self.F = F

    def forward(self, x_d):

        dk_t = self.F * x_d[:,0] * sqrt(pi * x_d[:,1])
        dk_t = dk_t[:,None]

        return dk_t


class DkNN(nn.Module):
    def __init__(self,Strain, atrain, **kwargs):
        super(DkNN, self).__init__(**kwargs)
        # stress-intensity layer
        self.dKlayer = nn.Sequential(
        Normalization(min(Strain), max(Strain), min(atrain), max(atrain)),
        nn.Linear(2, 5),
        nn.Tanh(),
        nn.Linear(5, 1)
        )

    def forward(self, x):

        dk_t = self.dKlayer(x)

        return dk_t


def dk_training_loop(n_epochs,batch_size, optimizer, model, loss_fn, train, label):
    for epoch in range(1, n_epochs + 1):
        
        permutation = randperm(train.shape[0])
        for i in range(0,train.shape[0], batch_size):

            #Minibatch
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = train[indices], label[indices]
            
            #Forward pass
            output_train = model(batch_x)
            loss_train = loss_fn(output_train, batch_y)

            #Backward pass
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

        print(f"{int(i/batch_size)+1}/{batch_size}-Epoch {epoch}, Training loss {loss_train.item():.4e}")

def hPINN_training_loop(n_epochs, optimizer, model, loss_fn, train, a0, label):
    for epoch in range(1, n_epochs + 1):        
            #Forward pass
            output_train = model(train, a0)
            loss_train = loss_fn(output_train, label)

            #Backward pass
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            
            print(f"Epoch {epoch}, Training loss {loss_train.item():.4e}")


if __name__ == "__main__":
    # Paris law coefficients
    [C, m] = [1.5E-11, 3.8]
    F = 1.0
    
    # data
    Strain = np.asarray(pd.read_csv('./data/Strain.csv'))[:,:,np.newaxis]
    Strain = Tensor(Strain)
    atrain = np.asarray(pd.read_csv('./data/atrain.csv'))
    atrain = Tensor(atrain)
    a0 = np.asarray(pd.read_csv('./data/a0.csv'))[0,0]*np.ones((Strain.shape[0],1))
    a0 = Tensor(a0)

    "-------------------------------------------------------------------------"
    "PINN with DkPhys "
    dk_phys = DkPhys(F)
    euler = EulerIntegratorCell(C=C, m=m, dKlayer=dk_phys)
    model_PINN = MyRNN(cell = euler)
    aPred = model_PINN(Strain, a0)[:, :]

    "-------------------------------------------------------------------------"
    " dKlayerNN and weight initialization "
    S_range = linspace(min(Strain), max(Strain), 1000)
    a_range = linspace(min(atrain), max(atrain), 1000)[randperm(1000)]
    dK_range = -12.05 + 0.24 * S_range + 760.0 * a_range
    dK_range = dK_range [:,None]

    dKlayer = DkNN(Strain,atrain)

    dk_training_loop(
        n_epochs = 100,
        batch_size = 32,
        optimizer = optim.RMSprop(dKlayer.parameters(), lr=1e-2),
        model = dKlayer,
        loss_fn = nn.MSELoss(),
        train = transpose(stack([S_range,a_range]),0,1),
        label = dK_range
        )

    "-------------------------------------------------------------------------"
    " hPINN training "
    h_euler = EulerIntegratorCell(C=C, m=m, dKlayer=dKlayer)
    model_hPINN = MyRNN(cell = h_euler)
    aPred_before = model_hPINN(Strain, a0)[:,:]

    hPINN_training_loop(
        n_epochs = 100,
        optimizer = optim.RMSprop(model_hPINN.parameters(), lr=1e-2),
        model = model_hPINN,
        loss_fn = nn.MSELoss(),
        train = Strain,
        a0 = a0,
        label = atrain,
    )

    h_aPred = model_PINN(Strain, a0)[:,:]

    "-------------------------------------------------------------------------"
    " plottting "
    matplotlib.rc('font', size=12)

    fig, ax= plt.subplots(1,2,sharey=True, figsize=(2*6.4,4.8),gridspec_kw={'hspace': 0, 'wspace': 0.1}) 

    ax[0].plot([0,0.05],[0,0.05],'--k')
    ax[0].plot(atrain.detach().numpy(), aPred_before.detach().numpy(), 'o', label = 'before training')
    ax[0].plot(atrain.detach().numpy(), h_aPred.detach().numpy(), 's', label = 'after training')
    ax[0].set_xlabel("actual crack length (m)")
    ax[0].set_ylabel("predicted crack length (m)")
    ax[0].legend(loc = 'upper left',facecolor = 'w')
    ax[0].grid(which='both')

    ax[1].plot([0,0.05],[0,0.05],'--k')
    ax[1].plot(atrain.detach().numpy(), aPred.detach().numpy(), 's', label = 'physical $\Delta$K')
    ax[1].grid(which='both')
    ax[1].set_xlabel("actual crack length (m)")
    ax[1].legend(loc = 'upper left',facecolor = 'w')

    plt.show()