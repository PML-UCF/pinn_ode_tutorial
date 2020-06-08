from tensorflow.keras.callbacks import ModelCheckpoint
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
    # data
    df = pd.read_csv('data.csv')
    t = df[['t']].values
    utrain = df[['u0', 'u1']].values
    ytrain = df[['yT0', 'yT1']].values
    utrain = utrain[np.newaxis, :, :]
    ytrain = ytrain[np.newaxis, :, :]

    x0 = np.zeros((2 * n,), dtype='float32')
    initial_state = [x0[np.newaxis, :]]

    # Callback
    mckp = ModelCheckpoint(filepath="./savedmodels/cp.ckpt", monitor='loss', verbose=1,
                           save_best_only=True, mode='min', save_weights_only=True)

    # fitting physics-informed neural network
    model = create_model(m, c, k, dt, batch_input_shape=utrain.shape, initial_state=initial_state,
                         return_sequences=True, unroll=False)
    history = model.fit(utrain, ytrain, epochs=100, steps_per_epoch=1, verbose=1, callbacks=[mckp])

    # plotting predictions
    fig = plt.figure()
    plt.plot(np.array(history.history['loss']))
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(which='both')
    plt.show()
