from tensorflow.keras.layers import RNN, Dense, Layer
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.python.framework import ops, tensor_shape
from tensorflow.python.ops import array_ops
import pandas as pd
import numpy as np
from tensorflow import float32


class EulerIntegratorCell(Layer):
    def __init__(self, C, m, dKlayer, units=1, a0=None, **kwargs):
        super(EulerIntegratorCell, self).__init__(**kwargs)
        self.units = units
        self.C = C
        self.m = m
        self.a0 = a0
        self.dKlayer = dKlayer
        self.state_size = tensor_shape.TensorShape(self.units)

    def build(self, input_shape, **kwargs):
        self.built = True

    def call(self, inputs, states):
        inputs = ops.convert_to_tensor(inputs)
        a_tm1 = ops.convert_to_tensor(states)

        x_d_tm1 = array_ops.concat((inputs, a_tm1[0, :]), axis=1)
        dk_t = self.dKlayer(x_d_tm1)
        da_t = self.C * (dk_t ** self.m)
        a = da_t + a_tm1[0, :]
        return a, [a]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.a0


class Normalization(Layer):
    def __init__(self, S_low, S_up, a_low, a_up, **kwargs):
        super(Normalization, self).__init__(**kwargs)
        self.low_bound_S = S_low
        self.upper_bound_S = S_up
        self.low_bound_a = a_low
        self.upper_bound_a = a_up

    def build(self, input_shape, **kwargs):
        self.built = True

    def call(self, inputs):
        delta_x = inputs - [self.low_bound_S, self.low_bound_a]
        output = delta_x / [(self.upper_bound_S - self.low_bound_S), (self.upper_bound_a - self.low_bound_a)]
        return output


def create_model(C, m, dKlayer, a0, batch_input_shape, return_sequences):
    euler = EulerIntegratorCell(batch_input_shape=batch_input_shape,
                                C=C, m=m, dKlayer=dKlayer,
                                a0=a0)

    PINN = RNN(cell=euler, batch_input_shape=batch_input_shape, return_sequences=return_sequences,
               return_state=False)

    model = Sequential()
    model.add(PINN)
    model.compile(loss='mse', optimizer=RMSprop(1e-2))

    return model


"-------------------------------------------------------------------------"
# Building up dK layer MLP

C = 1.5E-11  # Paris model constant
m = 3.8  # Paris model exponent

Strain = np.asarray(pd.read_csv('Strain.csv'))[:, :, np.newaxis]
Stest = np.asarray(pd.read_csv('Stest.csv'))[:, :, np.newaxis]
atrain = np.asarray(pd.read_csv('atrain.csv'))
a0 = np.asarray(pd.read_csv('a0.csv'))
a0 = ops.convert_to_tensor(a0, dtype=float32)

dKlayer = Sequential()
dKlayer.add(Normalization(np.min(Strain), np.max(Strain), np.min(atrain), np.max(atrain)))
dKlayer.add(Dense(5, activation='tanh'))
dKlayer.add(Dense(1))

S_range = np.linspace(np.min(Strain), np.max(Strain), 1000)
a_range = np.linspace(np.min(atrain), np.max(atrain), 1000)
dK_range = np.linspace(0, 40.0, 1000)  # Estimated dKlayer range

dKlayer.compile(loss='mse', optimizer=Adam(5e-1))
inputs_train = np.transpose(np.asarray([S_range, a_range]))
dKlayer.fit(inputs_train, dK_range, epochs=5)

model = create_model(C, m, dKlayer, a0, batch_input_shape=Strain.shape, return_sequences=False)
model.fit(Strain, atrain, epochs=30, steps_per_epoch=1, verbose=1)
model_predict = create_model(C, m, dKlayer, a0, batch_input_shape=Stest.shape, return_sequences=True)
model_predict.set_weights(model.get_weights())
aPred = model_predict.predict_on_batch(Stest)[:, :, 0]
