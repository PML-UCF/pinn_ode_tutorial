import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.framework import ops
from tensorflow import float32
from model import Normalization, create_model

if __name__ == "__main__":
    # Paris law coefficients
    [C, m] = [1.5E-11, 3.8]
    
    # data
    Strain = np.asarray(pd.read_csv('./data/Strain.csv'))[:, :, np.newaxis]
    atrain = np.asarray(pd.read_csv('./data/atrain.csv'))
    a0     = np.asarray(pd.read_csv('./data/a0.csv'))[0,0]*np.ones((Strain.shape[0],1))
    
    # stress-intensity layer
    dKlayer = Sequential()
    dKlayer.add(Normalization(70, 160, 0.005, 0.03))
    dKlayer.add(Dense(5, activation='tanh'))
    dKlayer.add(Dense(1))

    # weight initialization
    S_range  = np.linspace(70, 160, 1000)
    a_range  = np.linspace(0.005, 0.03, 1000)[np.random.permutation(np.arange(1000))]
    dK_range = -12.05 + 0.24 * S_range + 760.0 * a_range

    dKlayer.compile(loss='mse', optimizer=RMSprop(1e-2))
    inputs_train = np.transpose(np.asarray([S_range, a_range]))
    dKlayer.fit(inputs_train, dK_range, epochs=100)

    # fitting physics-informed neural network
    mckp = ModelCheckpoint(filepath = "./savedmodels/cp.ckpt", monitor = 'loss', verbose = 1,
                           save_best_only = True, mode = 'min', save_weights_only = True)
    
    model = create_model(C=C, m=m, a0=ops.convert_to_tensor(a0, dtype=float32), dKlayer=dKlayer, batch_input_shape=Strain.shape)
    history = model.fit(Strain, atrain, epochs=100, steps_per_epoch=1, verbose=1, callbacks=[mckp])

    # plotting predictions
    fig = plt.figure()
    plt.plot(np.array(history.history['loss']))
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(which='both')
    plt.show()
    