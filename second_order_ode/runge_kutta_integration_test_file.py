import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as matplotlib


def runga_kutta_vibrations(t, z1_0, z2_0, v1_0, v2_0, m1, m2, c1, c2, c3, k1, k2, k3, u1, u2):

    # Defining displacement and velocity matrices
    z1 = np.zeros(t.shape)
    z2 = np.zeros(t.shape)
    v1 = np.zeros(t.shape)
    v2 = np.zeros(t.shape)
    z1[0] = z1_0
    z2[0] = z2_0
    v1[0] = v1_0
    v2[0] = v2_0
    dt = t[1] - t[0]

    # Functions to calculate the acceleration a
    def a1_func(z1, z2, v1, v2, u1):
        return (u1 - (c1+c2) * v1 + c2 * v2 - (k1+k2) * z1 + k2 * z2) / m1

    def a2_func(z1, z2, v1, v2, u2):
        return (u2 - (c3+c2) * v2 + c2 * v1 - (k3+k2) * z2 + k2 * z1) / m2

    # Runge-Kutta integration
    for i in range(t.size - 1):
        # u at time step t / 2
        u1_t_05 = (u1[i + 1] - u1[i]) / 2 + u1[i]
        u2_t_05 = (u2[i + 1] - u2[i]) / 2 + u2[i]

        # Step k=1
        z1_1 = z1[i]
        v1_1 = v1[i]

        z2_1 = z2[i]
        v2_1 = v2[i]

        a1_1 = a1_func(z1_1, z2_1, v1_1, v2_1, u1[i])
        a2_1 = a2_func(z1_1, z2_1, v1_1, v2_1, u2[i])

        # Step k=2
        z1_2 = z1[i] + v1_1 * dt / 2
        v1_2 = v1[i] + a1_1 * dt / 2

        z2_2 = z2[i] + v2_1 * dt / 2
        v2_2 = v2[i] + a2_1 * dt / 2

        a1_2 = a1_func(z1_2, z2_2, v1_2, v2_2, u1_t_05)
        a2_2 = a2_func(z1_2, z2_2, v1_2, v2_2, u2_t_05)

        # Step k=3
        z1_3 = z1[i] + v1_2 * dt / 2
        v1_3 = v1[i] + a1_2 * dt / 2

        z2_3 = z2[i] + v2_2 * dt / 2
        v2_3 = v2[i] + a2_2 * dt / 2

        a1_3 = a1_func(z1_3, z2_3, v1_3, v2_3, u1_t_05)
        a2_3 = a2_func(z1_3, z2_3, v1_3, v2_3, u2_t_05)

        # Step k=4
        z1_4 = z1[i] + v1_3 * dt
        v1_4 = v1[i] + a1_3 * dt

        z2_4 = z2[i] + v2_3 * dt
        v2_4 = v2[i] + a2_3 * dt

        a1_4 = a1_func(z1_4, z2_4, v1_4, v2_4, u1[i+1])
        a2_4 = a2_func(z1_4, z2_4, v1_4, v2_4, u2[i+1])

        # Calculate displacement and velocity for next time step
        z1[i + 1] = z1[i] + dt / 6 * (v1_1 + 2 * v1_2 + 2 * v1_3 + v1_4)
        z2[i + 1] = z2[i] + dt / 6 * (v2_1 + 2 * v2_2 + 2 * v2_3 + v2_4)

        v1[i + 1] = v1[i] + dt / 6 * (a1_1 + 2 * a1_2 + 2 * a1_3 + a1_4)
        v2[i + 1] = v2[i] + dt / 6 * (a2_1 + 2 * a2_2 + 2 * a2_3 + a2_4)

    return z1, z2, v1, v2


if __name__ == "__main__":
    # Defining mass, stiffness and damping coefficients
    m1 = 20.0
    m2 = 10.0

    k1 = 2.0e3
    k2 = 1.0e3
    k3 = 5.0e3

    c1 = 100.0
    c2 = 110.0
    c3 = 120.0

    df = pd.read_csv('data.csv')
    t = df[['t']].values
    u1 = df[['u0']].values
    u2 = df[['u1']].values

    z1_0 = 0.0
    z2_0 = 0.0
    v1_0 = 0.0
    v2_0 = 0.0

    u1_test = u1[1]
    u2_test = u2[0]
    t_test = t[1]

    z1, z2, v1, v2 = runga_kutta_vibrations(t, z1_0, z2_0, v1_0, v2_0, m1, m2, c1, c2, c3, k1, k2, k3, u1, u2)

    # plot displacement over time
    matplotlib.rc('font', size=14)

    y1 = df[['y0']].values
    y2 = df[['y1']].values

    ifig = 1
    fig = plt.figure(ifig)
    fig.clf()

    plt.plot(t, z1, 'b', label='z1')
    plt.plot(t, z2, 'r', label='z2')
    plt.plot(t, y1, 'grey')
    plt.plot(t, y2, 'grey')
    plt.xlabel('t [s]')
    plt.ylabel('z [m]')
    plt.grid()
    plt.legend()
    plt.show()
