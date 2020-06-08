from tensorflow.keras.layers import RNN, Dense, Layer
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.ops import array_ops, gen_math_ops
import tensorflow as tf


class RungeKuttaIntegratorCell(Layer):
    def __init__(self, m, c, k, dt, initial_state, input_n_dim, units=1, **kwargs):
        super(RungeKuttaIntegratorCell, self).__init__(**kwargs)
        self.units = units
        self.m = m
        self.c = c
        self.k = k
        self.initial_state = initial_state
        self.state_size = 2 * input_n_dim
        self.dt = dt

    def build(self, input_shape, **kwargs):

        self.C = self.add_weight("C", shape = self.c.shape, trainable = True, initializer = lambda shape, dtype: self.c, **kwargs)
        self.M = self.add_weight("M", shape = self.m.shape, trainable = False, initializer = lambda shape, dtype: self.m, **kwargs)
        self.K = self.add_weight("K", shape = self.k.shape, trainable = False, initializer = lambda shape, dtype: self.k, **kwargs)

        self.built = True

    def call(self, inputs, states):

        u = inputs
        x = states[0]

        # Runge-Kutta integration
        # u at time step t / 2
        u_t_05 = u

        # Step k=1
        z_1 = array_ops.concat(([[x[0, 0]]], [[x[0, 2]]]), axis=1)
        v_1 = array_ops.concat(([[x[0, 1]]], [[x[0, 3]]]), axis=1)
        a_1 = self.a_func(z_1, v_1, u)

        # Step k=2
        z_2 = z_1 + v_1 * self.dt / 2
        v_2 = v_1 + a_1 * self.dt / 2
        a_2 = self.a_func(z_2, v_2, u_t_05)

        # Step k=3
        z_3 = z_1 + v_2 * self.dt / 2
        v_3 = v_1 + a_2 * self.dt / 2
        a_3 = self.a_func(z_3, v_3, u_t_05)

        # Step k=4
        z_4 = z_1 + v_3 * self.dt
        v_4 = v_1 + a_3 * self.dt
        a_4 = self.a_func(z_4, v_4, u)

        # Calculate displacement and velocity for next time step
        x_0 = z_1[0, 0] + self.dt / 6 * (v_1[0, 0] + 2 * v_2[0, 0] + 2 * v_3[0, 0] + v_4[0, 0])
        x_1 = v_1[0, 0] + self.dt / 6 * (a_1[0, 0] + 2 * a_2[0, 0] + 2 * a_3[0, 0] + a_4[0, 0])
        x_2 = z_1[0, 1] + self.dt / 6 * (v_1[0, 1] + 2 * v_2[0, 1] + 2 * v_3[0, 1] + v_4[0, 1])
        x_3 = v_1[0, 1] + self.dt / 6 * (a_1[0, 1] + 2 * a_2[0, 1] + 2 * a_3[0, 1] + a_4[0, 1])

        y = array_ops.concat(([[x_0]], [[x_2]]), axis=-1)
        x = tf.convert_to_tensor([[x_0, x_1, x_2, x_3]])

        return y, [x]

    def a_func(self, z, v, u):
        a1 = (u[0, 0] - (self.C[0] + self.C[1]) * v[0, 0] + self.C[1] * v[0, 1] - (self.K[0] + self.K[1]) * z[0, 0] + self.K[1] * z[0, 1]) / self.M[0]
        a2 = (u[0, 1] - (self.C[2] + self.C[1]) * v[0, 1] + self.C[1] * v[0, 0] - (self.K[2] + self.K[1]) * z[0, 1] + self.K[1] * z[0, 0]) / self.M[1]
        return array_ops.concat(([[a1]], [[a2]]), axis=1)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.initial_state


def create_model(m, c, k, dt, batch_input_shape, initial_state = None, return_sequences = True, unroll = False):
    input_n_dim = batch_input_shape[-1]
    rkCell = RungeKuttaIntegratorCell(m=m, c=c, k=k, dt=dt, initial_state=initial_state, input_n_dim=input_n_dim)
    ssRNN = RNN(cell=rkCell, batch_input_shape=batch_input_shape, return_sequences=True, return_state=False, unroll=False)

    model = Sequential()
    model.add(ssRNN)
    model.compile(loss='mse', optimizer=RMSprop(1e4), metrics=['mae'])
    return model
