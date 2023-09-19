import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre, hermite, roots_legendre, roots_hermite
import sympy as sp

### NOTE: THIS DOES NOT WORK YET - SOME PROBLEM WITH THE KE MATRIX. ###


def tridiag(npoints, k1=-1, k2=0, k3=1):
    a = np.ones(npoints - 1)*1
    b = np.ones(npoints)*-2
    c = np.ones(npoints - 1)*1
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

class Basis:

    def __init__(self, degree, x, wfunc):
        self.degree = degree
        self.ngrid = len(x)
        self.x = x
        self.dx = x[1] - x[0]
        self.w = self._eval_weight_function(wfunc)
        self.poly = None
        self.deriv = None
        self.psi = None
        self.Xmat = None
        self.U = None
        self.x_quad = None
        self.weights = None
        self.dvr_funcs = None
        self.theta = None

    def dvr_transform(self):
        if not self.poly.any():
            raise Exception('Basis Polynomials not initialised.')
        self.psi = self._fbr()
        self.Xmat = self._coordinate_matrix()
        self.U, self.x_quad = self._diagonalize_coord_matrix()
        self.weights = self._calculate_weights()
        xq, self.weights_exact = roots_legendre(self.degree)
        self.x_quad_exact = xq * np.eye(self.degree)
        self.theta, self.dvr_funcs = self._calculate_dvr_functions()

    def _eval_weight_function(self, wfunc):
        return wfunc(self.x)

    def _fbr(self):
        psi = np.zeros((self.degree, self.ngrid))
        for n in range(self.degree):
            psi[n, :] = np.sqrt(self.w) * self.poly[n, :]
        return psi

    def _coordinate_matrix(self):
        degree = self.degree
        Xmat = np.zeros((degree, degree))
        for i in range(degree):
            for j in range(degree):
                Xmat[i, j] = np.trapz(self.psi[i, :] * (self.x * self.psi[j, :]), self.x)
        #Itd = self._tridiag(degree)
        #Xmat *= Itd
        return Xmat

    def _diagonalize_coord_matrix(self):
        vals, vecs = np.linalg.eigh(self.Xmat)
        U = vecs
        Uinv = np.linalg.inv(U)
        x_quad = Uinv @ self.Xmat @ U
        return U, x_quad

    def _calculate_weights(self):
        weights = np.zeros(self.degree)
        Uinv = np.linalg.inv(self.U)
        for n in range(self.degree):
            weights[n] = Uinv[n, 0]**2 * np.trapz(self.w, dx=self.dx)
        return weights

    def _calculate_dvr_functions(self):
        degree, ngrid = self.degree, self.ngrid
        S = np.zeros((degree, ngrid))
        Uinv = np.linalg.inv(self.U)
        for i in range(degree):
            for j in range(degree):
                S[i, :] += Uinv[i, j] * self.psi[j, :]
            phase = np.sign(np.average(S[i, :]))
            S[i, :] *= phase
        dvr_funcs = np.zeros((degree, ngrid))
        for i in range(degree):
            dvr_funcs[i, :] = (self.weights[i] / (self.w**0.5))**0.5 * S[i, :]
            phase = np.sign(np.average(dvr_funcs[i, :]))
            dvr_funcs[i, :] *= phase
        return S, dvr_funcs

    @staticmethod
    def _tridiag(npoints, k1=-1, k2=0, k3=1):
        a = np.ones(npoints-1)
        b = np.ones(npoints)
        c = np.ones(npoints-1)
        return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

    def calculate_overlap_matrix(self, P, x):
        I = np.zeros((self.degree, self.degree))
        for i in range(self.degree):
            for j in range(self.degree):
                integrand = np.trapz(P[i, :] * P[j, :], x)
                I[i, j] = integrand
        return I

class Hermite(Basis):

    def __init__(self, degree, x, transform=False):
        super().__init__(degree, x, self._wfunc)
        self.poly, self.d1, self.d2 = self._hermite_polynomial(self.ngrid, self.x)
        if transform:
            super().dvr_transform()

    def _wfunc(self, x):
        return np.exp(-x**2)

    def eval_at_quadpoints(self):
        self.poly, self.d1, self.d2 = self._hermite_polynomial(self.degree, np.diag(self.x_quad))

    def _hermite_polynomial(self, m, x):
        polynomails = np.zeros((self.degree, m))
        poly_d1 = np.zeros((self.degree, m))
        poly_d2 = np.zeros((self.degree, m))
        for n in range(self.degree):
            poly = hermite(n)
            d1 = poly.deriv(1)
            d2 = poly.deriv(2)
            p_n = np.polyval(poly, x)
            norm = np.sqrt(np.trapz(p_n**2, x))
            p_n /= norm
            polynomails[n, :] = p_n
            p_d1 = np.polyval(d1, x)
            p_d2 = np.polyval(d2, x)
            poly_d1[n, :] = p_d1
            poly_d2[n, :] = p_d2
        return polynomails, poly_d1, poly_d2


class Legendre(Basis):

    def __init__(self, degree, x, transform=False):
        super().__init__(degree, x, self._wfunc)
        self.poly, self.d1, self.d2 = self._legendre_polynomial(self.ngrid, self.x)
        if transform:
            super().dvr_transform()

    def _wfunc(self, x):
        return np.ones(len(x))

    def eval_at_quadpoints(self):
        self.poly, self.d1, self.d2 = self._legendre_polynomial(self.degree, np.diag(self.x_quad))

    def _legendre_polynomial(self, m, x):
        polynomails = np.zeros((self.degree, m))
        poly_d1 = np.zeros((self.degree, m))
        poly_d2 = np.zeros((self.degree, m))
        for n in range(self.degree):
            poly = legendre(n)
            d1 = poly.deriv(1)
            d2 = poly.deriv(2)
            p_n = np.polyval(poly, x)
            norm = np.sqrt(np.trapz(p_n**2, x))
            p_n /= norm
            polynomails[n, :] = p_n
            polynomails[n, 0] = 0
            polynomails[n, -1] = 0
            p_d1 = np.polyval(d1, x)
            p_d2 = np.polyval(d2, x)
            poly_d1[n, :] = p_d1
            poly_d2[n, :] = p_d2
            poly_d2 /= norm
        return polynomails, poly_d1, poly_d2

def harmonic_oscillator(x, k):
    v = 0.5 * k * x**2
    return v

def calculate_second_derivative(f_values, h):
    n = len(f_values)
    second_derivatives = np.zeros(n)

    for i in range(n):
        if i == 0:
            second_derivatives[i] = (f_values[i + 2] - 2 * f_values[i + 1] + f_values[i]) / h ** 2
            second_derivatives[i] = 0
        elif i == n - 1:
            second_derivatives[i] = (f_values[i] - 2 * f_values[i - 1] + f_values[i - 2]) / h ** 2
            second_derivatives[i] = 0
        else:
            second_derivatives[i] = (f_values[i + 1] - 2 * f_values[i] + f_values[i - 1]) / h ** 2

    return second_derivatives

def hermite_harmonic():
     hbar, mass = 1, 1
     xmin, xmax = -20, 20
     ngrid = 100001
     degree = 12
     x = np.linspace(xmin, xmax, ngrid)
     basis = Hermite(degree, x)
     basis.dvr_transform()
     v = harmonic_oscillator(np.diag(basis.x_quad), k=1)
     V = np.eye(degree) * v
     breakpoint()
     basis.eval_at_quadpoints()

     Uinv = np.linalg.inv(basis.U)
     Vh = basis.U @ V @ Uinv

     Theg = np.zeros((degree, degree))
     for m in range(degree):
         for n in range(degree):
             for i in range(degree):
                 Theg[m, n] +=  basis.weights[i] * (basis.poly[m, i] * basis.d2[n, i]
                               + 2 * basis.d1[m, i] * basis.d1[n, i]
                               + basis.poly[n, i] * basis.d2[m, i])

     Theg *= -(hbar**2) / (2*mass)

     H = Theg + Vh

     E, c = np.linalg.eigh(H)
     breakpoint()

def legendre_harmonic():
    hbar, mass = 1, 1
    xmin, xmax = -1, 1
    ngrid = 10001
    degree = 12
    xgrid = np.linspace(xmin, xmax, ngrid)
    basis = Legendre(degree, xgrid)
    basis.dvr_transform()
    v = harmonic_oscillator(np.diag(basis.x_quad), k=100000)
    vb = harmonic_oscillator(xgrid, k=100000)
    V = np.eye(degree) * v
    xd = np.diag(basis.x_quad)
    #basis.eval_at_quadpoints()

    Uinv = np.linalg.inv(basis.U)
    Vh = basis.U @ V @ Uinv

    Vb = np.zeros((degree, degree))
    for i in range(degree):
        for j in range(degree):
            Vb[i, j] = np.trapz(basis.theta[i, :] * (vb * basis.theta[j, :]), xgrid)
            #Vb[i, j] = np.inner(basis.theta[j, :], (vb * basis.theta[i, :]))


    S = np.zeros((degree, degree))
    for i in range(degree):
        for j in range(degree):
            S[i, j] = np.trapz(basis.theta[i, :] * basis.theta[j, :], xgrid)

    D = tridiag(ngrid)
    #D[0, 0], D[0, 1], D[0, 2] = 1, -2, 1
    #D[-1, -1], D[-1, -2], D[-1, -3] = 1, -2, 1

    xd_min, xd_max = np.min(xd), np.max(xd)

    x, m, n = sp.symbols('x m n')
    derivs = np.zeros((degree, ngrid))
    polys = np.zeros((degree, ngrid))
    for n in range(degree):
        pn = sp.legendre(n, x)
        fp = sp.lambdify(x, pn, "numpy")
        pv = fp(xgrid)
        N = sp.integrate(pn**2, (x, -1, 1))
        N = np.sqrt(float(N))
        polys[n, :] = pv / N
        if n <= 2:
            derivs[n, :] = 0
        else:
            dp = sp.diff(pn, x, 2)
            f = sp.lambdify(x, dp, "numpy")
            dv = f(xgrid)
            derivs[n, :] = dv / N


    theta = np.zeros((degree, ngrid))
    theta_deriv = np.zeros((degree, ngrid))
    Uinv = np.linalg.inv(basis.U)
    for i in range(degree):
        for j in range(degree):
            theta[i, :] += Uinv[i, j] * polys[j, :]
            theta_deriv[i, :] += Uinv[i, j] * derivs[j, :]
        #theta[i, :] = (basis.weights[i] / (basis.w ** 0.5)) ** 0.5 * theta[i, :]
        #theta_deriv[i, :] = (basis.weights[i] / (basis.w ** 0.5)) ** 0.5 * theta_deriv[i, :]
        phase = np.sign(np.average(theta[i, :]))
        #theta[i, :] *= phase
        #theta_deriv[i, :] *= phase

    ind = 0
    #ind = 903
    #ind = 573

    ind = np.argmin(np.abs(xgrid - np.diag(basis.x_quad)[0]))

    T = np.zeros((degree, degree))
    D2 = np.zeros((degree-2, int(len(xgrid[ind:ngrid-ind]))))
    for i in range(degree):
        for j in range(1, degree-1):
            #d2 = calculate_second_derivative(basis.theta[j, ind:ngrid-ind], h=x[1]-x[0])
            #D2[j, :] = d2
            #T[i, j] = np.trapz(basis.theta[i, ind:ngrid-ind] * d2, dx=x[1]-x[0])
            #doverlap = 0
            #for n in range(degree):
            #    d2 = calculate_second_derivative(basis.psi[n, ind:ngrid-ind], h=x[1] - x[0])
            #    doverlap += np.inner(basis.psi[n, ind:ngrid-ind], d2[:])
            d2 = calculate_second_derivative(basis.dvr_funcs[j, ind:ngrid-ind], h=xgrid[1] - xgrid[0])
            D2[j-1, :] = d2
            T[i, j-1] = np.trapz(basis.dvr_funcs[i, ind:ngrid-ind] * D2[j-1, :], xgrid[ind:ngrid-ind])
            T[i, j-1] *= -(hbar**2)/(2*mass)


    #T *= (-hbar**2)/(2*mass)
    #T = Uinv @ T @ basis.U
    #Td = T[3:-3, 3:-3]
    #Vd = V[3:-3, 3:-3]
    H = T[1:-1, 1:-1] + V[1:-1, 1:-1]

    E, c = np.linalg.eig(H)
    sort_vals = E.argsort()
    E.sort()
    c = c[:, sort_vals]
    breakpoint()



if __name__ == "__main__":
    legendre_harmonic()
