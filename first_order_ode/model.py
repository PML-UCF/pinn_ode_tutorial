from tensorflow.keras.layers import RNN, Layer
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.framework import ops, tensor_shape
from tensorflow.python.ops import array_ops

class EulerIntegratorCell(Layer):
    def __init__(self, C, m, dKlayer, a0=None, units=1, **kwargs):
        super(EulerIntegratorCell, self).__init__(**kwargs)
        self.units = units
        self.C     = C
        self.m     = m
        self.a0    = a0
        self.dKlayer     = dKlayer
        self.state_size  = tensor_shape.TensorShape(self.units)
        self.output_size = tensor_shape.TensorShape(self.units)

    def build(self, input_shape, **kwargs):
        self.built = True

    def call(self, inputs, states):
        inputs  = ops.convert_to_tensor(inputs)
        a_tm1   = ops.convert_to_tensor(states)
        x_d_tm1 = array_ops.concat((inputs, a_tm1[0, :]), axis=1)
        dk_t    = self.dKlayer(x_d_tm1)
        da_t    = self.C * (dk_t ** self.m)
        a       = da_t + a_tm1[0, :]
        return a, [a]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.a0

class Normalization(Layer):
    def __init__(self, S_low, S_up, a_low, a_up, **kwargs):
        super(Normalization, self).__init__(**kwargs)
        self.low_bound_S   = S_low
        self.upper_bound_S = S_up
        self.low_bound_a   = a_low
        self.upper_bound_a = a_up

    def build(self, input_shape, **kwargs):
        self.built = True

    def call(self, inputs):
        output  = (inputs - [self.low_bound_S, self.low_bound_a]) / [(self.upper_bound_S - self.low_bound_S), (self.upper_bound_a - self.low_bound_a)]
        return output

def create_model(C, m, a0, dKlayer, batch_input_shape, return_sequences=False, return_state=False):
    euler = EulerIntegratorCell(C=C, m=m, dKlayer=dKlayer, a0=a0, batch_input_shape=batch_input_shape)
    PINN  = RNN(cell=euler, batch_input_shape=batch_input_shape, return_sequences=return_sequences, return_state=return_state)
    model = Sequential()
    model.add(PINN)
    model.compile(loss='mse', optimizer=RMSprop(1e-2))
    return model
