from tensorflow.keras.layers import RNN, Dense, Layer
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.framework import ops, tensor_shape
from tensorflow.python.ops import array_ops
from pandas import read_csv
from numpy import asarray, newaxis
from tensorflow import float32


class EulerIntegratorCell(Layer):
    def __init__(self,  C, m, dKlayer, units=1, a0=None, **kwargs):
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
        initial_state = self.a0
        return initial_state

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

Strain = asarray(read_csv('Strain.csv'))
atrain = asarray(read_csv('atrain.csv'))
a0 = asarray(read_csv('a0.csv'))
Stest = asarray(read_csv('Stest.csv'))

Strain = Strain[:, :, newaxis]
a0 = ops.convert_to_tensor(a0, dtype=float32)
Stest = Stest[:, :, newaxis]


dKlayer = Sequential()
dKlayer.add(Dense(5, activation='tanh'))
dKlayer.add(Dense(1))

model = create_model(C, m, dKlayer, a0, batch_input_shape=Strain.shape, return_sequences=False)
model.fit(Strain, atrain, epochs=5, steps_per_epoch=1, verbose=1)
aPred = model.predict_on_batch(Stest)[:, :]
