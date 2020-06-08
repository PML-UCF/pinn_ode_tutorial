from tensorflow.keras.layers import RNN, Dense, Layer
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.ops import array_ops, gen_math_ops
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class RungeKuttaIntegratorCell(Layer):
    def __init__(self, m, c, k, dt, initial_state, **kwargs):
        super(RungeKuttaIntegratorCell, self).__init__(**kwargs)
        self.M = m
        self.c = c
        self.K = k
        self.initial_state = initial_state
        self.state_size = 2 * len(m) # twice the number of degrees of freedom
        self.dt = dt

    def build(self, input_shape, **kwargs):
        self.C = self.add_weight("C", shape = self.c.shape, trainable = True,
                                 initializer = lambda shape, dtype: self.c, **kwargs)
        self.built = True

    def call(self, inputs, states):
        u = inputs
        x = states[0]

        # Runge-Kutta integration (assuming u(t+dt/2) = u(t))

        # Step k=1
        y_1     = array_ops.concat(([[x[0, 0]]], [[x[0, 2]]]), axis=1)
        ydot_1  = array_ops.concat(([[x[0, 1]]], [[x[0, 3]]]), axis=1)
        yddot_1 = self._fun(y_1, ydot_1, u)

        # Step k=2
        y_2 = y_1    + ydot_1  * self.dt / 2.
        ydot_2 = ydot_1 + yddot_1 * self.dt / 2.
        yddot_2 = self._fun(y_2, ydot_2, u)

        # Step k=3
        y_3 = y_1 + ydot_2 * self.dt / 2.
        ydot_3 = ydot_1 + yddot_2 * self.dt / 2.
        yddot_3 = self._fun(y_3, ydot_3, u)

        # Step k=4
        y_4 = y_1 + ydot_3 * self.dt
        ydot_4 = ydot_1 + yddot_3 * self.dt
        yddot_4 = self._fun(y_4, ydot_4, u)

        # Calculate displacement and velocity for next time step
        x_0 = y_1[0, 0] + self.dt / 6 * (ydot_1[0, 0] + 2 * ydot_2[0, 0] + 2 * ydot_3[0, 0] + ydot_4[0, 0])
        x_2 = y_1[0, 1] + self.dt / 6 * (ydot_1[0, 1] + 2 * ydot_2[0, 1] + 2 * ydot_3[0, 1] + ydot_4[0, 1])
        x_1 = ydot_1[0, 0] + self.dt / 6 * (yddot_1[0, 0] + 2 * yddot_2[0, 0] + 2 * yddot_3[0, 0] + yddot_4[0, 0])
        x_3 = ydot_1[0, 1] + self.dt / 6 * (yddot_1[0, 1] + 2 * yddot_2[0, 1] + 2 * yddot_3[0, 1] + yddot_4[0, 1])

        y = array_ops.concat(([[x_0]], [[x_2]]), axis=-1)
        x = tf.convert_to_tensor([[x_0, x_1, x_2, x_3]])

        return y, [x]

    def _fun(self, z, v, u):
        a1 = (u[0, 0] - (self.C[0] + self.C[1]) * v[0, 0] + self.C[1] * v[0, 1] - (self.K[0] + self.K[1]) * z[0, 0] + self.K[1] * z[0, 1]) / self.M[0]
        a2 = (u[0, 1] - (self.C[2] + self.C[1]) * v[0, 1] + self.C[1] * v[0, 0] - (self.K[2] + self.K[1]) * z[0, 1] + self.K[1] * z[0, 0]) / self.M[1]
        return array_ops.concat(([[a1]], [[a2]]), axis=1)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.initial_state


def create_model(m, c, k, dt, initial_state, batch_input_shape, return_sequences = True, unroll = False):
    rkCell = RungeKuttaIntegratorCell(m=m, c=c, k=k, dt=dt, initial_state=initial_state)
    ssRNN = RNN(cell=rkCell, batch_input_shape=batch_input_shape, return_sequences=return_sequences, return_state=False, unroll=unroll)
    model = Sequential()
    model.add(ssRNN)
    model.compile(loss='mse', optimizer=RMSprop(1e4), metrics=['mae'])
    return model

if __name__ == "__main__":
    # mass, spring coefficient, damping coefficient
    m = np.array([20.0, 10.0])
    c = np.array([10.0, 10.0, 10.0]) # initial guess
    k = np.array([2e3, 1e3, 5e3])

    # data
    df = pd.read_csv('data.csv')
    t  = df[['t']].values
    dt = (t[1] - t[0])[0]
    utrain = df[['u0', 'u1']].values
    utrain = utrain[np.newaxis, :, :]
    ytrain = df[['yT0', 'yT1']].values
    ytrain = ytrain[np.newaxis, :, :]

    initial_state = np.zeros((1,2 * len(m),), dtype='float32')

    # fitting physics-informed neural network
    model = create_model(m, c, k, dt, initial_state=initial_state, batch_input_shape=utrain.shape)
    yPred_before = model.predict_on_batch(utrain)[0, :, :]
#    model.fit(utrain, ytrain, epochs=1, steps_per_epoch=1, verbose=1)
#    yPred = model.predict_on_batch(utrain)[0, :, :]

    # plotting prediction results
    plt.plot(t, ytrain[0, :, :], 'gray')
    plt.plot(t, yPred_before[:, :], 'r', label='before training')
#    plt.plot(t, yPred[:, :], 'b', label='after training')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.grid('on')
    plt.legend()
    plt.show()
