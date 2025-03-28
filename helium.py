import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.special import sph_harm
from scipy.special import genlaguerre
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline
from math import factorial
import os
from tabulate import tabulate


r_max = 30 # radial boundaries
N = 100


def setup_grid(N):

    P = np.zeros(N + 1) # polynomial coefficients
    P[-1] = 1 
    dP_dx = np.polynomial.legendre.legder(P) # derivative coefficients
    x = np.zeros(N + 1) # grid points
    x[0] = -1 
    x[N] = 1
    x[1: -1] = np.polynomial.legendre.legroots(dP_dx) # polynomial derivative roots
    P_x = np.polynomial.legendre.legval(x, P) # evaluating polynomial at grid points
    r = r_max/2 * (x + 1) # linear mapping
    dr_dx = r_max/2 * np.ones_like(r)
    
    return(x[1: -1], r[1: -1], dr_dx[1: -1], P_x[1: -1])


def D_matrix(N, x, dr_dx):

    D = np.zeros((N - 1, N - 1))

    for i in range(N - 1):

        for j in range(N - 1):

            if i == j:
                D[i][j] = (-1/3)*N*(N + 1)/(1 - x[i]**2)/dr_dx[i]**2
            else:
                D[i][j] = -2/(x[i] - x[j])**2/(dr_dx[i]*dr_dx[j])

    return D


def potential(l, r, x, dr_dx, P_x, C):

    V = -2/r + direct_potential(l, r, x, dr_dx, P_x, C)
    
    return V


def direct_potential(l, r, x, dr_dx, P_x, C):

    D = np.zeros((N - 1, N - 1))
    B = np.zeros((N - 1, N - 1))
    gll_weights = 2/(N*(N + 1)*P_x**2)
    V_grid_bound = (r/r_max)**2 * dr_dx * gll_weights # boundary
    D_bound = -2/(x - 1)**2 # boundary

    for i in range(N - 1):

        for j in range(N - 1):

            B[i][j] = -(2*l + 1)/r[j] - D_bound[i]*V_grid_bound[j]

            if i == j:
                D[i][j] = (-1/3)*N*(N + 1)/(1 - x[i]**2) - l*(l + 1)/r[i]**2
                
            else:
                D[i][j] = -2/(x[i] - x[j])**2

    V_grid = LA.inv(D) @ B
    V_contracted = np.sum(np.conj(C) * C * V_grid.T, axis = 0)
    V_direct = 4*np.pi/(2*l+1) * V_contracted / r

    return V_direct


def setup_matrix(N, l, r, D, C):

    V = potential(l, r, x, dr_dx, P_x, C)
    M = np.zeros((N - 1, N - 1))
    d = V + l*(l + 1)/(2*r**2) # values on the diagonal
    np.fill_diagonal(M, d)
    M += -0.5*D

    return M


def radial(N, l, C):

    M = setup_matrix(N, l, r, D, C)

    E, eigf = LA.eigh(M)
    print(E[0])

    f = eigf.T * P_x
    u = f / np.sqrt(dr_dx)
    R = u / r
    R_norm = np.zeros((N - 1, N - 1))

    for i in range(len(f)):

        gll_weights = 2/(N*(N + 1)*P_x**2)
        integral = np.sum(gll_weights * f[i]**2)
        R_norm[i] = R[i] / np.sqrt(integral)
        
    return(R_norm, u[0], E[0])


x, r, dr_dx, P_x = setup_grid(N)
D = D_matrix(N, x, dr_dx)

C = np.zeros(N - 1) # starting guess
E_0 = 1 # dummy
E_0_updated = 0 # dummy

while abs(E_0 - E_0_updated) > 1e-8:

    E_0 = E_0_updated
    R_norm, u_0, E_0_updated = radial(N, 0, C)
    C = u_0
    



"""
def radial_analytical(l, n, r):

    norm_c = np.sqrt((2 / n) ** 3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
    lag_pol = genlaguerre(n - l - 1, 2 * l + 1)(2 * r / n)
    R = norm_c * np.exp(-r / n) * (2 * r / n) ** l * lag_pol

    return R


def plot_radial(N, r_max, n_max):
    
    l_list = list(range(n_max))

    plt.figure()

    for l in l_list:
        r, R = radial(N, l)

        for i in range(l, n_max):
            n = i + 1

            plt.plot(r[1: -1], -R[i], label = f"{n}{l}")

            R_an = radial_analytical(l, n, r[1:-1])
            plt.plot(r[1: -1], R_an, label = f"{n}{l} analytical")

            plt.xlim(0, 5)
            #plt.xlim(0, r_max)
            plt.grid()
            plt.legend()
            plt.savefig("3d_hydrogen_radial.png")


def plot_error(N, r_max, n_max):

    l_list = list(range(n_max))

    plt.figure()

    for l in l_list:
        r, R = radial(N, l)

        for i in range(l, n_max):
            n = i + 1

            R_an = radial_analytical(l, n, r[1: -1])

            plt.semilogy(r[1: -1], np.abs(np.abs(R_an) - np.abs(R[i])), label = f"{n}{l} error")

            plt.xlim(0, r_max)
            plt.grid()
            plt.legend()
            plt.savefig("error.png")


def plot_pdf(N, n_max):

    l_list = list(range(n_max))

    x_mesh, y_mesh, z_mesh = np.mgrid[-17 : 17 : (N - 1)*1j, -17 : 17 : (N - 1)*1j, -17 : 17 : (N - 1)*1j]

    r_int = np.sqrt(x_mesh**2 + y_mesh**2 + z_mesh**2)
    phi = np.arctan2(np.sqrt(x_mesh**2 + y_mesh**2), z_mesh)
    theta = np.arctan2(y_mesh, x_mesh)

    for l in l_list:
        r, R = radial(N, l)

        for m in range(-l, l + 1):
            Y = sph_harm(m, l, theta, phi)

            for i in range(l, n_max):
                cs = CubicSpline(r[1: -1], R[i])
                R_int_n = cs(r_int)
               
                pdf = R_int_n ** 2 * np.abs(Y) ** 2
                
                fig = go.Figure(data=go.Volume(
                x=x_mesh.flatten(),
                y=y_mesh.flatten(),
                z=z_mesh.flatten(),
                value=pdf.flatten(),
                #isomin=0.1,
                #isomax=0.8,
                opacity=0.05, 
                surface_count=100,
                ))
                
                fig.write_image(os.path.join("plots", f"hydrogen_pdf_{i + 1}{l}{m}.png"))
                fig.write_html(os.path.join("htmls", f"hydrogen_pdf_{i + 1}{l}{m}.html"))
"""


