from tensorflow.keras.layers import RNN, Layer
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.ops import array_ops, gen_math_ops
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class RungeKuttaIntegratorCell(Layer):
    def __init__(self, m, c, k, dt, A, B, initial_state, **kwargs):
        super(RungeKuttaIntegratorCell, self).__init__(**kwargs)
        self.c = c
        self.initial_state = initial_state
        self.state_size = 2 * len(m)  # twice the number of degrees of freedom
        self.dt = dt
        self.A = A
        self.B = tf.dtypes.cast(B, tf.float32)
        self.K = tf.dtypes.cast(tf.convert_to_tensor(self._getCKmatrix(k)), tf.float32)
        self.Minv = tf.dtypes.cast(tf.linalg.inv(np.diag(m)), tf.float32)

    def build(self, input_shape, **kwargs):
        self.C_train = self.add_weight("C", shape = self.c.shape, trainable = True, initializer = lambda shape, dtype: self.c, **kwargs)
        self.built = True

    def call(self, inputs, states):

        C = tf.dtypes.cast(self._getCKmatrix(self.C_train), tf.float32)
        u = inputs
        x = states[0]
        y = x[:, :2]
        ydot = x[:, 2:]

        ui = tf.dtypes.cast(u, tf.float32)
        yddoti = self._fun(self.Minv, self.K, C, ui, y, ydot)
        yi = None
        ydoti = None
        fn = None
        for j in range(4):
            yn = y + self.A[j] * ydot * self.dt
            ydotn = ydot + self.A[j] * yddoti * self.dt
            if j == 0:
                yi = yn
                ydoti = ydotn
                fn = self._fun(self.Minv, self.K, C, ui, yn, ydotn)
            else:
                yi = tf.concat([yi, yn], axis=0)
                ydoti = tf.concat([ydoti, ydotn], axis=0)
                fn = tf.concat([fn, self._fun(self.Minv, self.K, C, ui, yn, ydotn)], axis=0)

        y = y + tf.matmul(self.B, ydoti) * self.dt
        ydot = ydot + tf.matmul(self.B, fn) * self.dt
        x = array_ops.concat(([y, ydot]), axis=-1)

        return y, [x]

    def _fun(self, Minv, K, C, ui, yi, ydoti):
        return tf.linalg.matmul(ui - tf.linalg.matmul(ydoti, C, transpose_b=True) - tf.linalg.matmul(yi, K, transpose_b=True), Minv, transpose_b=True)

    def _getCKmatrix(self, a):
        return [[a[0] + a[1], -a[1]], [-a[1], a[1] + a[2]]]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.initial_state


def create_model(m, c, k, dt, A, B, initial_state=None, batch_input_shape=None, return_sequences = True, unroll = False):
    rkCell = RungeKuttaIntegratorCell(m=m, c=c, k=k, dt=dt, A=A, B=B, initial_state=initial_state)
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

    A = np.array([0., 0.5, 0.5, 1.0])
    B = np.array([[1/6, 2/6, 2/6, 1/6]])

    # data
    df = pd.read_csv('data.csv')
    t = df[['t']].values
    dt = (t[1] - t[0])[0]
    utrain = df[['u0', 'u1']].values[np.newaxis, :, :]
    ytrain = df[['yT0', 'yT1']].values[np.newaxis, :, :]

    initial_state = np.zeros((1,2 * len(m),), dtype='float32')

    # fitting physics-informed neural network
    model = create_model(m, c, k, dt, A, B, initial_state=initial_state, batch_input_shape=utrain.shape)
    yPred_before = model.predict_on_batch(utrain)[0, :, :]
    model.fit(utrain, ytrain, epochs=100, steps_per_epoch=1, verbose=1)
    yPred = model.predict_on_batch(utrain)[0, :, :]

    # plotting prediction results
    plt.plot(t, ytrain[0, :, :], 'gray')
    plt.plot(t, yPred_before[:, :], 'r', label='before training')
    plt.plot(t, yPred[:, :], 'b', label='after training')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.grid('on')
    plt.legend()
    plt.show()
