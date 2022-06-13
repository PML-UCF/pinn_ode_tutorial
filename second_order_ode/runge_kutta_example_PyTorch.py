import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as matplotlib

from torch.nn.parameter import Parameter
from torch import (
    linalg,
    nn,
    Tensor,
    stack,
    cat,
    transpose, 
    optim,
    zeros,
    diag
    )


class MyRNN(nn.Module):
    def __init__(self, cell, **kwargs):
        super(MyRNN, self).__init__()
        self.cell = cell

    def forward(self, inputs, initial_state):

        bs, seq_sz, _ = inputs.shape
        state = []
        state.append(initial_state)
        for t in range(1, seq_sz): 
            input = inputs[:, t-1, :]
            state_t = self.cell.forward(input, state[t-1])
            state.append(state[t-1]+state_t)

        return stack((state),dim=1)


class RungeKuttaIntegratorCell(nn.Module):
    def __init__(self, m, c, k, dt, **kwargs):
        super(RungeKuttaIntegratorCell, self).__init__(**kwargs)
        self.Minv = linalg.inv(diag(m))
        self.c1 = Parameter(c[0])
        self.c2 = Parameter(c[1])
        self.c3 = Parameter(c[2])
        
        self.K    = Tensor([[k[0]+k[1],-k[1]],[-k[1],k[1]+k[2]]])
        self.state_size    = 2*len(m)
        self.A  = Tensor([0., 0.5, 0.5, 1.0])
        self.B  = Tensor([[1/6, 2/6, 2/6, 1/6]])
        self.dt = dt
        
    def forward(self, inputs, states):
        C = stack((stack((self.c1+self.c2, -self.c2)), stack((-self.c2, self.c2+self.c3))))
        y    = states[:, :2]
        ydot = states[:, 2:]
        
        yddoti = self._fun(self.Minv, self.K, C, inputs, y, ydot)
        yi     = y + self.A[0] * ydot * self.dt
        ydoti  = ydot + self.A[0] * yddoti * self.dt
        fn     = self._fun(self.Minv, self.K, C, inputs, yi, ydoti)
        for j in range(1,4):
            yn    = y + self.A[j] * ydot * self.dt
            ydotn = ydot + self.A[j] * yddoti * self.dt
            ydoti = cat([ydoti, ydotn], dim=0)
            fn    = cat([fn, self._fun(self.Minv, self.K, C, inputs, yn, ydotn)], dim=0)

        y    = linalg.matmul(self.B, ydoti) * self.dt
        ydot =  linalg.matmul(self.B, fn) * self.dt

        return cat(([y, ydot]), dim=-1)

    def _fun(self, Minv, K, C, u, y, ydot):
        return linalg.matmul(u - linalg.matmul(ydot, transpose(C, 0, 1)) - linalg.matmul(y, transpose (K, 0, 1)), transpose(Minv, 0, 1))

    
def pinn_training_loop(n_epochs, optimizer, model, loss_fn, train, label, initial_state):
    mae = nn.L1Loss()
    for epoch in range(1, n_epochs + 1):
        #Forward pass
        output_train = model(train, initial_state)[:, :, :2]
        loss_train = loss_fn(output_train, label)
        mae_train = mae(output_train, label)

        #Backward pass
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Training loss {loss_train.item():.4e}, mae {mae_train.item():.4e}")


if __name__ == "__main__":
    # masses, spring coefficients, and damping coefficients
    m = Tensor([20.0, 10.0])
    k = Tensor([2e3, 1e3, 5e3])
    c = Tensor([10.0, 10.0, 10.0]) # initial guess for damping coefficient


    # data
    df = pd.read_csv('./data/data.csv')
    t  = df[['t']].values
    dt = (t[1] - t[0])[0]
    utrain = df[['u0', 'u1']].values[np.newaxis, :, :]
    ytrain = df[['yT0', 'yT1']].values[np.newaxis, :, :]
    t = Tensor(t)
    utrain = Tensor(utrain)
    ytrain = Tensor(ytrain)

    # Initial state of the system 
    initial_state = zeros((1,2 * len(m)))

    rkCell = RungeKuttaIntegratorCell(m=m, c=c, k=k, dt=dt)
    model = MyRNN(cell=rkCell)
    
    #prediction results before training
    yPred_before = model(utrain, initial_state)[0, :, :]
    yPred_before = yPred_before.detach().numpy()[:,:2]

    #PINN training
    pinn_training_loop(
        n_epochs = 100,
        optimizer = optim.RMSprop(model.parameters(), lr=1e4),
        model = model,
        loss_fn = nn.MSELoss(),
        train = utrain,
        label = ytrain,
        initial_state=initial_state
        )

    #prediction results after training
    yPred = model(utrain, initial_state) [0, :, :]
    yPred = yPred.detach().numpy()[:,:2]

    # plotting prediction results
    plt.plot(t, ytrain[0, :, :], 'gray')
    plt.plot(t, yPred_before, 'r', label='before training')
    plt.plot(t, yPred, 'b', label='after training')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.grid('on')
    plt.legend()
    plt.show()
