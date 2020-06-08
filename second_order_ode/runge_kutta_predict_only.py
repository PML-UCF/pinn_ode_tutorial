import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import create_model

if __name__ == "__main__":
    # mass, spring coefficient, damping coefficient
    m = np.array([20.0, 10.0])
    c = np.array([10.0, 10.0, 10.0])
    k = np.array([2e3, 1e3, 5e3])

    n = 2
    dt = 0.002

    # data
    df = pd.read_csv('data_test.csv')
    t = df[['t']].values
    utest = df[['u0', 'u1']].values
    ytarget = df[['y1', 'y2']].values
    utest = utest[np.newaxis, :, :]
    ytarget = ytarget[np.newaxis, :, :]

    x0 = np.zeros((2 * n,), dtype='float32')
    initial_state = [x0[np.newaxis, :]]

    # fitting physics-informed neural network
    model = create_model(m, c, k, dt, batch_input_shape=utest.shape, initial_state=initial_state, return_sequences=True, unroll=False)
    yPred_before = model.predict_on_batch(utest)[0, :, :]
    model.load_weights("./savedmodels/cp.ckpt")
    yPred = model.predict_on_batch(utest)[0, :, :]

    ifig = 1
    fig = plt.figure(ifig)
    fig.clf()

    plt.plot(t, yPred_before[:, :], 'r', label='before training')
    plt.plot(t, yPred[:, :], 'b', label='after training')
    plt.plot(t, ytarget[0, :, 0], 'grey', label='y_Target')
    plt.plot(t, ytarget[0, :, 1], 'grey')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.grid('on')
    plt.legend()
    plt.show()
