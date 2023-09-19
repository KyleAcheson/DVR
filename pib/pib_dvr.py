import numpy as np
import matplotlib.pyplot as plt

def harmonic_oscillator(x, k=1):
    v = 0.5 * k * x**2
    return v

def double_well(xgrid, a=10, b=0.25):
    y = a * (b * xgrid**4 - xgrid**2)
    y -= np.min(y)
    return y

def get_basis(x, nbasis, ngrid, x_min, L):
    theta = np.zeros((nbasis, ngrid))
    for i in range(nbasis):
            y_i = np.sqrt(2.0 / L) * np.sin(float(i+1) * np.pi * (x - x_min) / L)
            #theta[i, :] = np.sqrt(2.0 / Length) * np.sin(float(ii) * np.pi * (x - Xmin) / Length)
            theta[i, :] = y_i
    return theta

def pib_dvr(x, v, nbasis, hbar=1, mass=1, neig=4, fname=None):
    """
    Performs DVR using a basis of localised particle in a box functions.

    :param x: x values on which potential is evaluated
    :param v: potential values on the grid
    :param nbasis: number of pib basis functions
    :param hbar: reduced Planck constant - defaults to 1
    :param mass: defaults to 1
    :param neig: number of eignestates to plot/ return
    :param fname:  file name to save wf plots to
    :return: energies, basis function coeffs, pib basis functions
    """
    x_min, x_max = np.min(x), np.max(x)
    L = x_max - x_min
    ngrid = len(x)

    theta = get_basis(x, nbasis, ngrid, x_min, L)
    V = np.zeros((nbasis, nbasis))
    for i in range(nbasis):
        for j in range(nbasis):
            y_i = theta[i, :]
            y_j = theta[j, :]
            Vn = y_i * v * y_j
            V[i, j] = np.trapz(Vn, x)

    T = np.zeros((nbasis, nbasis))
    for i in range(nbasis):
        for j in range(nbasis):
            y_i = theta[i, :]
            y_j = theta[j, :]
            deriv = (float(j+1) * np.pi/ L)**2
            Tn = y_i * deriv * y_j
            T[i, j] = np.trapz(Tn, x)
            T[i, j] *= (hbar**2)/(2*mass)

    H = T + V
    E, c = np.linalg.eig(H)
    sort_vals = E.argsort()
    E.sort()
    c = c[:, sort_vals]

    if fname:
        for i in range(neig):
            y = c[:, i] @ theta[:, :]
            sign = np.sign(np.average(y))
            y *= sign
            if i % 2:
                ls = '-'
            else:
                ls = '--'
            plt.plot(x, E[i] + y, linestyle=ls, linewidth=2, alpha=0.7)

        plt.plot(x, v, '-k', linewidth=1)
        plt.savefig(fname)

    return E[0:neig], c[:, 0:neig], theta

if __name__ == "__main__":
    ngrid = 1000
    nbasis = 50
    x_min, x_max = -5, 5
    x = np.linspace(x_min, x_max, ngrid)

    #v = harmonic_oscillator(x, k=1)
    v = double_well(x, a=3, b=0.045)

    energies, coeffs, dvr_funcs = pib_dvr(x, v, nbasis, neig=4, fname='pib_dvr_test.png')

    breakpoint()
