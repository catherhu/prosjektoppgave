import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.special import sph_harm
from scipy.special import genlaguerre
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline
from math import factorial
import os
from utils import setup_grid
from utils import D2_matrix


def setup_matrix(l):

    V = -1/r

    H = np.zeros((N - 1, N - 1))
    d = np.zeros(N - 1)
    d = V + l*(l + 1)/(2*r**2)
    np.fill_diagonal(H, d)

    H += -0.5*D2

    return(H, r, dr_dx, P_x)


def solve(l):

    H, r, dr_dx, P_x = setup_matrix(l)

    E, eigf = LA.eigh(H)
    
    f = eigf.T * P_x
    u = f / np.sqrt(dr_dx)
    R = u / r

    R_norm = np.zeros((N - 1, N - 1))

    for i in range(len(f)):
        integral = np.sum(gll_weights * f[i]**2)
        R_norm[i] = R[i] / np.sqrt(integral)
        
    return(E[0])


r_max = 30
N = 100
x, r, dr_dx, P_x, gll_weights = setup_grid(N, r_max)
D2 = D2_matrix(N, x, dr_dx)

E = solve(l = 0)
print(E)








"""
def radial_analytical(l, n):

    norm_c = np.sqrt((2 / n) ** 3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
    lag_pol = genlaguerre(n - l - 1, 2 * l + 1)(2 * r / n)
    R = norm_c * np.exp(-r / n) * (2 * r / n) ** l * lag_pol

    return R
"""
"""
def plot_radial(r_max, n_max):
    
    l_list = list(range(n_max))

    plt.figure()

    for l in l_list:
        r, R = solve(N, l)

        for i in range(l, n_max):
            n = i + 1

            plt.plot(r, -R[i], label = f"{n}{l}")

            R_an = radial_analytical(l, n, r)
            plt.plot(r, R_an, label = f"{n}{l} analytical")

            plt.xlim(0, 5)
            #plt.xlim(0, r_max)
            plt.grid()
            plt.legend()
            plt.savefig("3d_hydrogen_radial.png")
"""
"""
def plot_error(r_max, n_max):

    l_list = list(range(n_max))

    plt.figure()

    for l in l_list:
        r, R = solve(N, l)

        for i in range(l, n_max):
            n = i + 1

            R_an = radial_analytical(l, n, r)

            plt.semilogy(r, np.abs(np.abs(R_an) - np.abs(R[i])), label = f"{n}{l} error")

            plt.xlim(0, r_max)
            plt.grid()
            plt.legend()
            plt.savefig("error.png")
"""
"""
def plot_pdf(n_max):

    l_list = list(range(n_max))

    x_mesh, y_mesh, z_mesh = np.mgrid[-17 : 17 : (N - 1)*1j, -17 : 17 : (N - 1)*1j, -17 : 17 : (N - 1)*1j]

    r_int = np.sqrt(x_mesh**2 + y_mesh**2 + z_mesh**2)
    phi = np.arctan2(np.sqrt(x_mesh**2 + y_mesh**2), z_mesh)
    theta = np.arctan2(y_mesh, x_mesh)

    for l in l_list:
        r, R = solve(l)

        for m in range(-l, l + 1):
            Y = sph_harm(m, l, theta, phi)

            for i in range(l, n_max):
                cs = CubicSpline(r, R[i])
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






    