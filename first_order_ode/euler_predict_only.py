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
    Stest = np.asarray(pd.read_csv('./data/Stest.csv'))[:, :, np.newaxis]
    atest = np.asarray(pd.read_csv('./data/atest.csv'))
    a0    = np.asarray(pd.read_csv('./data/a0.csv'))[0,0]*np.ones((Stest.shape[0],1))
    
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

    # building the model and predicting before "training"
    model = create_model(C=C, m=m, a0=ops.convert_to_tensor(a0, dtype=float32),
                         dKlayer=dKlayer, return_sequences=True, batch_input_shape=Stest.shape)
    aBefore = model.predict_on_batch(Stest)[:,:,0].numpy()
    
    # loading weights from trained model
    model.load_weights("./savedmodels/cp.ckpt")
    aAfter = model.predict_on_batch(Stest)[:,:,0].numpy()
    
    # plotting predictions
    fig = plt.figure()
    plt.plot([0,0.05],[0,0.05],'--k')
    plt.plot(atest[:,-1],aBefore[:,-1],'o', label = 'before training')
    plt.plot(atest[:,-1],aAfter[:,-1], 's', label = 'after training')
    plt.xlabel("actual crack length (m)")
    plt.ylabel("predicted crack length (m)")
    plt.legend(loc = 'upper center',facecolor = 'w')
    plt.grid(which='both')
    plt.show()
