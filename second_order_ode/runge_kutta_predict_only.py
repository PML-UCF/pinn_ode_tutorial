import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import create_model

if __name__ == "__main__":
    # masses, spring coefficients, and damping coefficients
    m = np.array([20.0, 10.0], dtype='float32')
    c = np.array([30.0, 5.0, 10.0], dtype='float32') # initial guess
    k = np.array([2e3, 1e3, 5e3], dtype='float32')

    # data
    df = pd.read_csv('./data/data.csv')
    t  = df[['t']].values
    dt = (t[1] - t[0])[0]
    utest = df[['u0', 'u1']].values[np.newaxis, :, :]

    initial_state = np.zeros((1,2 * len(m),), dtype='float32')

    # fitting physics-informed neural network
    model = create_model(m, c, k, dt, initial_state=initial_state, batch_input_shape=utest.shape)
    yPred_before = model.predict_on_batch(utest)[0, :, :]
    model.load_weights("./savedmodels/cp.ckpt")
    yPred = model.predict_on_batch(utest)[0, :, :]

    # plotting predictions
#    plt.plot(t, ytrain[0, :, :], 'gray')
    plt.plot(t, yPred_before[:, :], 'r', label='before training')
    plt.plot(t, yPred[:, :], 'b', label='after training')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.grid('on')
    plt.legend()
    plt.show()