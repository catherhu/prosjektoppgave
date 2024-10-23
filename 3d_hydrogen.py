import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.special import sph_harm
from scipy.special import laguerre
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline
from math import factorial
import os

# constants:
r_max = 20 # radial boundaries
L = 1 # mapping constant
alpha = 2*L/r_max # mapping constant

N = 50


def setup_grid(N):

    P = np.zeros(N + 1) # polynomial coefficients
    P[-1] = 1 
    dP_dx = np.polynomial.legendre.legder(P) # derivative coefficients

    x = np.zeros(N + 1) # grid points
    x[0] = -1 
    x[N] = 1
    x[1: -1] = np.polynomial.legendre.legroots(dP_dx) # polynomial derivative roots

    P_x = np.polynomial.legendre.legval(x, P) # evaluating polynomial at grid points

    r = L*(1 + x)/(1 - x + alpha) # radial grid points
    dr_dx = L*(2 + alpha)/(1 - x + alpha)**2

    return(x, r, dr_dx, P_x)


def setup_matrix(N, l):

    x, r, dr_dx, P_x = setup_grid(N)
    V = -1/r[1: -1]

    M = np.zeros((N + 1, N + 1))
    d = np.zeros(N + 1)
    d[1: -1] = V + l*(l + 1)/(2*r[1: -1]) # values on the diagonal
    np.fill_diagonal(M, d)

    D = np.zeros((N + 1, N + 1))

    for i in range(1, N):
        for j in range(1, N):
            if i == j:
                D[i][j] = (-1/3)*N*(N + 1)/(1 - x[i]**2)/dr_dx[i]**2
            else:
                D[i][j] = -2/(x[i] - x[j])**2/(dr_dx[i]*dr_dx[j])

    M += -0.5*D

    return(M[1: -1, 1: -1], r, dr_dx, P_x)


def radial(N, l):

    M, r, dr_dx, P_x = setup_matrix(N, l)

    E, eigf = LA.eigh(M)
    f = eigf.T * P_x[1: -1]
    u = f / np.sqrt(dr_dx[1: -1])

    R = u / r[1: -1]
    integral = np.sum(2 / (N * (N + 1) * P_x[1: -1] ** 2) * f**2)
    R_norm = R / np.sqrt(integral)

    return(r, R_norm)


def radial_analytical(l, n, r_max):

    a0 = 0.529
    r = np.linspace(0, r_max, 100)

    norm_const = np.sqrt(((2 / (n * a0)) ** 3 * factorial(n - l - 1)) / (2 * n * factorial(n + l) ** 3))
    laguerre_pol = laguerre(n - l - 1, 2 * l + 1)(2 * r / (n * a0))
    R = norm_const * np.exp(-r / (n * a0)) * (2 * r / (n * a0))**l * laguerre_pol

    return(r, R)

def plot_radial(r_max, n_max):
    
    l_list = list(range(n_max))

    for l in l_list:
        r, R = radial(N, l)

        for i in range(l, n_max):

            n = i + 1
            R_n = R[i]
            plt.plot(r[1: -1], -R_n, label = f"{n}{l}")

            r_an, R_n_an = radial_analytical(l, n, r_max)
            plt.plot(r_an, R_n_an, label = f"{n}{l} analytical")

            plt.xlim(0, r_max)
            plt.ylim(-0.02, 0.15)
            plt.grid()
            plt.legend()
            plt.savefig("3d_hydrogen_radial.png")


plot_radial(r_max, n_max = 3)





    