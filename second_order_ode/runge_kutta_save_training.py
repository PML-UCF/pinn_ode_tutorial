from tensorflow.keras.callbacks import ModelCheckpoint
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
    utrain = df[['u0', 'u1']].values[np.newaxis, :, :]
    ytrain = df[['yT0', 'yT1']].values[np.newaxis, :, :]

    initial_state = np.zeros((1,2 * len(m),), dtype='float32')

    # Callback
    mckp = ModelCheckpoint(filepath="./savedmodels/cp.ckpt", monitor='loss', verbose=1,
                           save_best_only=True, mode='min', save_weights_only=True)

    # fitting physics-informed neural network
    model = create_model(m, c, k, dt, initial_state=initial_state, batch_input_shape=utrain.shape)
    history = model.fit(utrain, ytrain, epochs=20, steps_per_epoch=1, verbose=1, callbacks=[mckp])

    # plotting predictions
    fig = plt.figure()
    plt.plot(np.array(history.history['loss']))
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(which='both')
    plt.show()
