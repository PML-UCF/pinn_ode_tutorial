from tensorflow.keras.layers import RNN, Dense, Layer
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.ops import array_ops, gen_math_ops
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as matplotlib


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

        self.c_1 = self.add_weight("c_1", shape = self.c[0].shape, trainable = True, initializer = lambda shape, dtype: self.c[0], **kwargs)
        self.c_2 = self.add_weight("c_2", shape = self.c[1].shape, trainable = True, initializer = lambda shape, dtype: self.c[1], **kwargs)
        self.c_3 = self.add_weight("c_3", shape = self.c[2].shape, trainable = True, initializer = lambda shape, dtype: self.c[2], **kwargs)

        self.m_1 = self.add_weight("m_1", shape = self.m[0].shape, trainable = False, initializer = lambda shape, dtype: self.m[0], **kwargs)
        self.m_2 = self.add_weight("m_2", shape = self.m[1].shape, trainable = False, initializer = lambda shape, dtype: self.m[1], **kwargs)

        self.k_1 = self.add_weight("k_1", shape = self.k[0].shape, trainable = False, initializer = lambda shape, dtype: self.k[0], **kwargs)
        self.k_2 = self.add_weight("k_2", shape = self.k[1].shape, trainable = False, initializer = lambda shape, dtype: self.k[1], **kwargs)
        self.k_3 = self.add_weight("k_3", shape = self.k[2].shape, trainable = False, initializer = lambda shape, dtype: self.k[2], **kwargs)

        self.built = True

    def call(self, inputs, states):

        u = inputs
        x = states[0]

        # Runge-Kutta integration
        # u at time step t / 2
        u1_t_05 = u[0, 0]
        u2_t_05 = u[0, 1]

        # Step k=1
        z1_1 = x[0, 0]
        v1_1 = x[0, 1]
        z2_1 = x[0, 2]
        v2_1 = x[0, 3]

        a1_1 = self.a1_func(z1_1, z2_1, v1_1, v2_1, u[0, 0])
        a2_1 = self.a2_func(z1_1, z2_1, v1_1, v2_1, u[0, 1])

        # Step k=2
        z1_2 = x[0, 0] + v1_1 * self.dt / 2
        v1_2 = x[0, 1] + a1_1 * self.dt / 2
        z2_2 = x[0, 2] + v2_1 * self.dt / 2
        v2_2 = x[0, 3] + a2_1 * self.dt / 2

        a1_2 = self.a1_func(z1_2, z2_2, v1_2, v2_2, u1_t_05)
        a2_2 = self.a2_func(z1_2, z2_2, v1_2, v2_2, u2_t_05)

        # Step k=3
        z1_3 = x[0, 0] + v1_2 * self.dt / 2
        v1_3 = x[0, 1] + a1_2 * self.dt / 2
        z2_3 = x[0, 2] + v2_2 * self.dt / 2
        v2_3 = x[0, 3] + a2_2 * self.dt / 2

        a1_3 = self.a1_func(z1_3, z2_3, v1_3, v2_3, u1_t_05)
        a2_3 = self.a2_func(z1_3, z2_3, v1_3, v2_3, u2_t_05)

        # Step k=4
        z1_4 = x[0, 0] + v1_3 * self.dt
        v1_4 = x[0, 1] + a1_3 * self.dt
        z2_4 = x[0, 2] + v2_3 * self.dt
        v2_4 = x[0, 3] + a2_3 * self.dt

        a1_4 = self.a1_func(z1_4, z2_4, v1_4, v2_4, u[0, 0])
        a2_4 = self.a2_func(z1_4, z2_4, v1_4, v2_4, u[0, 1])

        # Calculate displacement and velocity for next time step
        x_0 = z1_1 + self.dt / 6 * (v1_1 + 2 * v1_2 + 2 * v1_3 + v1_4)
        x_2 = z2_1 + self.dt / 6 * (v2_1 + 2 * v2_2 + 2 * v2_3 + v2_4)
        x_1 = v1_1 + self.dt / 6 * (a1_1 + 2 * a1_2 + 2 * a1_3 + a1_4)
        x_3 = v2_1 + self.dt / 6 * (a2_1 + 2 * a2_2 + 2 * a2_3 + a2_4)

        y = array_ops.concat(([[x_0]], [[x_2]]), axis=-1)
        x = tf.convert_to_tensor([[x_0, x_1, x_2, x_3]])

        return y, [x]

    def a1_func(self, z1, z2, v1, v2, u1):
        return (u1 - (self.c_1 + self.c_2) * v1 + self.c_2 * v2 - (self.k_1 + self.k_2) * z1 + self.k_2 * z2) / self.m_1

    def a2_func(self, z1, z2, v1, v2, u2):
        return (u2 - (self.c_3 + self.c_2) * v2 + self.c_2 * v1 - (self.k_3 + self.k_2) * z2 + self.k_2 * z1) / self.m_2

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.initial_state


def create_model(m, c, k, dt, batch_input_shape, initial_state = None, return_sequences = True, unroll = False):

    input_n_dim = batch_input_shape[-1]
    rkCell = RungeKuttaIntegratorCell(m=m, c=c, k=k, dt=dt, initial_state=initial_state, input_n_dim=input_n_dim)
    rkRNN = RNN(cell=rkCell, batch_input_shape=batch_input_shape, return_sequences=True, return_state=False, unroll=False)

    model = Sequential()
    model.add(rkRNN)
    model.compile(loss='mse', optimizer=RMSprop(1e4), metrics=['mae'])

    return model

if __name__ == "__main__":
    # mass, spring coefficient, damping coefficient
    m = np.array([20.0, 10.0])
    c = np.array([10.0, 10.0, 10.0])
    k = np.array([2e3, 1e3, 5e3])

    n = 2
    dt = 0.002

    # data
    df = pd.read_csv('data.csv')
    t = df[['t']].values
    utrain = df[['u0', 'u1']].values
    utest = df[['u0', 'u1']].values
    ytrain = df[['yT0', 'yT1']].values

    x0 = np.zeros((2 * n,), dtype='float32')

    utrain = utrain[np.newaxis, :, :]
    utest = utest[np.newaxis, :, :]
    ytrain = ytrain[np.newaxis, :, :]

    batch_input_shape = utrain.shape
    initial_state = [x0[np.newaxis, :]]

    # fitting physics-informed neural network
    model = create_model(m, c, k, dt, batch_input_shape=utrain.shape, initial_state=initial_state, return_sequences=True, unroll=False)
    yPred_before = model.predict_on_batch(utest)[0, :, :]
    model.fit(utrain, ytrain, epochs=100, steps_per_epoch=1, verbose=1)
    yPred = model.predict_on_batch(utest)[0, :, :]

    # Plotting prediction results
    matplotlib.rc('font', size=14)

    ifig = 1
    fig = plt.figure(ifig)
    fig.clf()

    plt.plot(t, ytrain[0, :, :], 'gray')
    plt.plot(t, yPred_before[:, :], 'r', label='before training')
    plt.plot(t, yPred[:, :], 'b', label='after training')
    plt.xlabel('t')
    plt.ylabel('z')
    plt.grid('on')
    plt.legend()
    plt.show()
