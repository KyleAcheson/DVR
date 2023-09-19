import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.integrals.quadrature import gauss_lobatto

def harmonic_oscillator(x, k=1):
    y = 0.5 * k * x**2
    return y

def double_well(xgrid, a=10, b=0.25):
    y = a * (b * xgrid**4 - xgrid**2)
    y -= np.min(y)
    return y

def lobatto_newton(n, xgrid, tol=1e-6, max_iter=100):
    """
    Calculates the lobatto nodes by analytically finding the derivative
    of the polynomial of degree n-1 and solving for the roots using
    newton-raphson method. May struggle to converge for high degrees.
    """
    x = sp.symbols('x')
    p_n = sp.legendre(n-1, x)
    dp_n = sp.diff(p_n, x)
    df = sp.diff(dp_n, x)
    func = sp.lambdify(x, dp_n, 'numpy')
    df_func = sp.lambdify(x, df, 'numpy')
    vals = func(xgrid)
    abs_diff = np.abs(vals - 0)
    inds = np.argwhere(abs_diff < 10)
    initial_guesses = xgrid[inds]

    roots = []
    weights = np.zeros(n)

    for guess in initial_guesses:
        x0 = guess[0]
        for i in range(max_iter):
            f_val = func(x0)
            df_val = df_func(x0)
            if abs(f_val) < tol:
                roots.append(float(x0))
                break
            x0 = x0 - f_val / df_val

    roots = np.array(roots)
    inds = np.unique(roots.round(decimals=2), return_index=True)[1]
    roots = roots[inds]
    roots = np.hstack([[-1], roots, [1]])
    if len(roots) != n:
        raise Exception("Failed to find all roots")

    for i in range(n):
        y = p_n.subs(x, roots[i])
        w = 2 / (n*(n-1) * y**2)
        weights[i] = float(w)

    return roots, weights

def lobatto_nodes(n):
    """
    Calculates the lobatto nodes by analytically finding the derivative
    of the polynomial of degree n-1 and analytically solving for
    the roots using sympy. Slow for high degrees.
    """
    nodes = np.zeros(n)
    weights = np.zeros(n)
    nodes[0] = -1

    x = sp.symbols('x')
    p_n = sp.legendre(n-1, x)
    dp_n = sp.diff(p_n, x)
    func = sp.lambdify(x, dp_n, 'numpy')
    roots_sym = sp.solve(dp_n, x)
    i = 1
    for root in roots_sym:
        nodes[i] = root.evalf()
        i += 1

    nodes[-1] = 1
    for i in range(n):
        y = p_n.subs(x, nodes[i])
        w = 2 / (n*(n-1) * y**2)
        weights[i] = float(np.real(w))

    sort_inds = nodes.argsort()
    nodes.sort()
    weights = weights[sort_inds]

    return nodes, weights

def lagrange_interpolation(x_values, y_values, interp_x):
    interpolated_values = []
    for x in interp_x:
        result = 0
        for i, xi in enumerate(x_values):
            term = y_values[i]
            for j, xj in enumerate(x_values):
                if i != j:
                    term *= (x - xj) / (xi - xj)
            result += term
        interpolated_values.append(result)
    return np.array(interpolated_values)

def lagrange_j(x, x_nodes, node_ind):
    """
    Gives the lagrange polynomial at position node_ind
    """
    ng = len(x)
    nnode = len(x_nodes)
    p = np.ones(ng)
    for k in range(nnode):
        x_node = x_nodes[k]
        if k != node_ind:
            p *= (x - x_node) / (x_nodes[node_ind] - x_node)
    return p

def calculate_derivative(basis_polys, nodes, weights):
    """
    Calculates analytical derivatives for the lobatto shape functions
    """
    npolys, nnodes = basis_polys[1:-1, :].shape
    derivs = np.zeros((nnodes, nnodes))
    for i in range(0, nnodes):
        d = 0
        for j in range(0, nnodes):
            if j != i:
                d += 1.0 / (nodes[i] - nodes[j])
        derivs[i, i] = d
    for i in range(0, nnodes-1):
        for j in range(i + 1, nnodes):
            d = 1 / (nodes[i] - nodes[j])
            for k in range(0, nnodes):
                if k != i and k != j:
                    d *= (nodes[j] - nodes[k]) / (nodes[i] - nodes[k])
            derivs[i, j] = d
            derivs[j, i] = -d * weights[j] / weights[i]
    return derivs


def lobatto_dvr(xgrid, v, degree, hbar=1, mass=1, fname=None, neig=4):
    """
    Performs Lobatto DVR as described in https://doi.org/10.1016/0009-2614(88)87322-6.

    :param xgrid: grid of x points on which potential is defined
    :param v: values of the potential
    :param degree: degree of the polynomial used in lobatto quadrature (number of basis functions = degree - 2)
    :param hbar: reduced Planck constant - defaults to atomic units
    :param mass: mass - defaults to atomic units
    :param fname: file name to save plot of wfs to
    :param neig: number of eigenstates to plot
    :return: energies, basis function coefficients, dvr basis functions
    """
    xmax, xmin = np.max(xgrid), np.min(xgrid)
    x_scale = 0.5 * (xmax - xmin)

    precision = 18
    nodes, weights = gauss_lobatto(degree, precision)
    nodes, weights = np.array(nodes, dtype=float), np.array(weights, dtype=float)
    nodes *= x_scale
    weights *= x_scale

    basis_polys = np.zeros((degree, degree))
    dvr_basis = np.zeros((degree, degree))
    for j in range(degree):
        basis_polys[j, :] = lagrange_j(nodes, nodes, j)
        dvr_basis[j, :] = 1 / (np.sqrt(weights[j])) * basis_polys[j, :]

    vn = lagrange_interpolation(xgrid, v, nodes)

    V = np.zeros((degree, degree))
    for i in range(0, degree):
        for j in range(0, degree):
            for k in range(0, degree):
                V[i, j] += weights[k] * dvr_basis[i, k] * vn[k] * dvr_basis[j, k]
    derivs = calculate_derivative(basis_polys, nodes, weights)

    T = np.zeros((degree, degree))
    for i in range(0, degree):
        for j in range(0, degree):
            for k in range(0, degree):
                T[i, j] += weights[k] * derivs[i, k] * derivs[j, k]

    nbas = degree - 2
    for i in range(nbas):
        for j in range(nbas):
            T[i, j] = T[i + 1, j + 1] / np.sqrt(weights[i + 1] * weights[j + 1])

    T = T[:-2, :-2]
    T *= (hbar ** 2) / (2 * mass)
    H = T + V[1:-1, 1:-1]

    E, c = np.linalg.eigh(H)
    sort_vals = E.argsort()
    E.sort()
    c = c[:, sort_vals]

    if fname:
        for i in range(neig):
            y = c[:, i] @ dvr_basis[1:-1, :]
            sign = np.sign(np.average(y))
            y *= sign
            if i % 2:
                ls = '-'
            else:
                ls = '--'
            plt.plot(nodes, E[i] + y, linestyle=ls, linewidth=2, alpha=0.7)

        plt.plot(nodes, vn, '-k', linewidth=1)
        plt.savefig(fname)

    return E[0:neig], c[:, 0:neig], dvr_basis[1:-1, :]


if __name__ == "__main__":
    xmin, xmax = -5, 5
    ngrid = 20
    xgrid = np.linspace(xmin, xmax, ngrid)
    degree = 45

    #v = harmonic_oscillator(xgrid, k=1)
    v = double_well(xgrid, a=3, b=0.045)

    energies, coeffs, basis_funcs = lobatto_dvr(xgrid, v, degree, neig=4, fname='lobatto_test.png')