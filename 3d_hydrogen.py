import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.special import sph_harm
from scipy.special import genlaguerre
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline
from math import factorial
import os

# constants:
r_max = 30 # radial boundaries
L = 1 # mapping constant
alpha = 2*L/r_max # mapping constant

N = 1000


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
    d[1: -1] = V + l * (l + 1) / (2 * r[1: -1] ** 2) # values on the diagonal
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
    print(E[0:3])
    f = eigf.T * P_x[1: -1]
    u = f / np.sqrt(dr_dx[1: -1])
    
    R = u / r[1: -1]

    R_norm = np.zeros((N - 1, N - 1))

    for i in range(len(f)):
        integral = np.sum(2 / (N * (N + 1) * P_x[1: -1] ** 2) * f[i]**2)
        R_norm[i] = R[i] / np.sqrt(integral)
        
    return(r, R_norm)


def radial_analytical(l, n, r_max):

    r = np.linspace(0, r_max, 100)

    norm_c = np.sqrt((2 / n) ** 3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
    lag_pol = genlaguerre(n - l - 1, 2 * l + 1)(2 * r / n)
    R = norm_c * np.exp(-r / n) * (2 * r / n) ** l * lag_pol

    return(r, R)


def plot_radial(N, r_max, n_max):
    
    l_list = list(range(n_max))

    for l in l_list:
        r, R = radial(N, l)

        for i in range(l, n_max):
            n = i + 1

            plt.plot(r[1: -1], R[i], label = f"{n}{l}")

            r_an, R_n_an = radial_analytical(l, n, r_max)
            plt.plot(r_an, R_n_an, label = f"{n}{l} analytical")

            plt.xlim(0, 5)
            #plt.xlim(0, r_max)
            plt.grid()
            plt.legend()
            plt.savefig("3d_hydrogen_radial.png")


def plot_pdf(N, n_max):

    l_list = list(range(n_max))

    x_mesh, y_mesh, z_mesh = np.mgrid[-18 : 18 : (N - 1)*1j, -18 : 18 : (N - 1)*1j, -18 : 18 : (N - 1)*1j]

    r_int = np.sqrt(x_mesh**2 + y_mesh**2 + z_mesh**2)
    phi = np.arctan2(np.sqrt(x_mesh**2 + y_mesh**2), z_mesh)
    theta = np.arctan2(y_mesh, x_mesh)

    for l in l_list:
        r, R = radial(N, l)

        for m in range(0, l + 1):
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


plot_radial(N, r_max, n_max = 3)
#plot_pdf(N, n_max = 3)





    