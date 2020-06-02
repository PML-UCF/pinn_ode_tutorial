from tensorflow.keras.layers import RNN, Dense, Layer
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.ops import array_ops, gen_math_ops
from tensorflow.python.framework import ops
import pandas as pd
import numpy as np
import tensorflow as tf


class RungeKuttaIntegratorCell(Layer):
    def __init__(self, m, c, k, dt, initial_state, input_n_dim, units=1, **kwargs):
        super(RungeKuttaIntegratorCell, self).__init__(**kwargs)
        self.units = units
        # self.M = tf.Variable(M, trainable=False)
        # self.C = tf.Variable(C, trainable=True)
        # self.K = tf.Variable(K, trainable=False)
        self.m = m
        self.c = c
        self.k = k
        self.initial_state = initial_state
        self.state_size = 2 * input_n_dim
        self.dt = dt

    def build(self, input_shape, **kwargs):

        # self.c_1 = self.add_weight("c_1", shape = self.c[0].shape, trainable = True, initializer = lambda shape, dtype: self.c[0], **kwargs)
        # self.c_2 = self.add_weight("c_2", shape = self.c[1].shape, trainable = True, initializer = lambda shape, dtype: self.c[1], **kwargs)
        # self.c_3 = self.add_weight("c_3", shape = self.c[2].shape, trainable = True, initializer = lambda shape, dtype: self.c[2], **kwargs)
        #
        # self.m_1 = self.add_weight("m_1", shape = self.m[0].shape, trainable = False, initializer = lambda shape, dtype: self.m[0], **kwargs)
        # self.m_2 = self.add_weight("m_2", shape = self.m[1].shape, trainable = False, initializer = lambda shape, dtype: self.m[1], **kwargs)
        #
        # self.k_1 = self.add_weight("k_1", shape = self.k[0].shape, trainable = False, initializer = lambda shape, dtype: self.k[0], **kwargs)
        # self.k_2 = self.add_weight("k_2", shape = self.k[1].shape, trainable = False, initializer = lambda shape, dtype: self.k[1], **kwargs)
        # self.k_3 = self.add_weight("k_3", shape = self.k[2].shape, trainable = False, initializer = lambda shape, dtype: self.k[2], **kwargs)

        # self.k_1 = tf.Variable(self.k[0], trainable=False)
        # self.k_2 = tf.Variable(self.k[1], trainable=False)
        # self.k_3 = tf.Variable(self.k[2], trainable=False)

        # self.M = tf.convert_to_tensor([[self.m_1, 0], [0, self.m_2]], dtype=tf.float32)
        # self.C = tf.convert_to_tensor([[self.c_1 + self.c_2, -self.c_2], [-self.c_2, self.c_2 + self.c_3]], dtype=tf.float32)
        # self.K = tf.convert_to_tensor([[self.k_1 + self.k_2, -self.k_2], [-self.k_2, self.k_2 + self.k_3]], dtype=tf.float32)

        self.M = self.add_weight("M", shape = (2, 2), trainable = False, initializer = lambda shape, dtype: [[self.m[0], 0], [0, self.m[1]]], **kwargs)
        self.C = self.add_weight("C", shape = (2, 2), trainable = True, initializer = lambda shape, dtype: [[self.c[0] + self.c[1], -self.c[1]], [-self.c[1], self.c[1] + self.c[2]]], **kwargs)
        self.K = self.add_weight("K", shape = (2, 2), trainable = False, initializer = lambda shape, dtype: [[self.k[0] + self.k[1], -self.k[1]], [-self.k[1], self.k[1] + self.k[2]]], **kwargs)

        n = self.M.shape[0]
        self.I = tf.eye(n, dtype=tf.float32) #dtype=2self.m.dtype
        self.Z = tf.zeros([n, n], dtype=tf.float32)

        self.Minv = tf.linalg.inv(self.M)
        self.negMinvK = -1.0 * tf.matmul(self.Minv, self.K)
        self.negMinvC = -1.0 * tf.matmul(self.Minv, self.C)

        self.A1row = tf.concat([self.Z, self.I], axis=1)
        self.A2row = tf.concat([self.negMinvK, self.negMinvC], axis=1)
        self.Ac = tf.concat([self.A1row, self.A2row], axis=0)

        self.Bc = tf.concat([self.Z, self.Minv], axis=0)
        self.Cc = tf.concat([self.I, self.Z], axis=1)
        self.Dc = self.Z

        self._C2D()
        self.built = True

    def call(self, inputs, states, training=None):

        if training is not None:
            self._C2D()

        u = inputs
        x = states[0]

        y = gen_math_ops.mat_mul(x, self.Cd, transpose_b=True) + gen_math_ops.mat_mul(u, self.Dd, transpose_b=True)
        x = gen_math_ops.mat_mul(x, self.Ad, transpose_b=True) + gen_math_ops.mat_mul(u, self.Bd, transpose_b=True)

        return y, [x]

    def _C2D(self):

        self.negMinvC = -1.0 * tf.matmul(self.Minv, self.C)

        self.A1row = tf.concat([self.Z, self.I], axis=1)
        self.A2row = tf.concat([self.negMinvK, self.negMinvC], axis=1)
        self.Ac = tf.concat([self.A1row, self.A2row], axis=0)

        # Build an exponential matrix
        em_upper = array_ops.concat((self.Ac, self.Bc), axis=1)
        em_lower = array_ops.concat((tf.zeros((self.Bc.shape[1], self.Ac.shape[0]), dtype=self.Ac.dtype),
                                     tf.zeros((self.Bc.shape[1], self.Bc.shape[1]), dtype=self.Ac.dtype)), axis=1)
        em = array_ops.concat((em_upper, em_lower), axis=0)
        ms = tf.linalg.expm(self.dt * em)
        ms = ms[:self.Ac.shape[0], :]  # dispose of the lower rows

        self.Ad = ms[:, 0:self.Ac.shape[1]]
        self.Bd = ms[:, self.Ac.shape[1]:]

        self.Cd = self.Cc
        self.Dd = self.Dc

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.initial_state


def create_model(m, c, k, dt, batch_input_shape, initial_state = None, return_sequences = True,
                         unroll = False):

    input_n_dim = batch_input_shape[-1]
    rkCell = RungeKuttaIntegratorCell(m=m, c=c, k=k, dt=dt, initial_state=initial_state, input_n_dim=input_n_dim)

    ssRNN = RNN(cell=rkCell,
                batch_input_shape=batch_input_shape,
                return_sequences=True,
                return_state=False,
                unroll=False)

    model = Sequential()
    model.add(ssRNN)
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
    model = create_model(m, c, k, dt, batch_input_shape=utrain.shape, initial_state=initial_state, return_sequences=True,
                         unroll=False)

    yPred_before = model.predict_on_batch(utest)[0, :, :]

    model.fit(utrain, ytrain, epochs=100, steps_per_epoch=1, verbose=1)
    yPred = model.predict_on_batch(utest)[0, :, :]
