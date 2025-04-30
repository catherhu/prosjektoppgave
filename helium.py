import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.special import sph_harm
from scipy.special import genlaguerre
import plotly.graph_objects as go
from sympy.physics.wigner import gaunt
from scipy.interpolate import CubicSpline
from math import factorial
import os
from tabulate import tabulate


def setup_grid():

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
    gll_weights = 2/(N*(N + 1)*P_x**2)
    
    return(x, r[1: -1], dr_dx, P_x[1: -1], gll_weights[1: -1])


def D1_matrix():

    D1 = np.zeros((N + 1, N + 1))

    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                D1[i][j] = 1/(x[i] - x[j])/np.sqrt(dr_dx[i]*dr_dx[j])

    D1[0][0] = -0.25*N*(N + 1)/dr_dx[0]
    D1[N][N] = -D1[0][0]/dr_dx[N]

    return D1


def D2_matrix():

    D2 = np.zeros((N - 1, N - 1))
    x_ = x[1: -1]
    dr_dx_ = dr_dx[1: -1]

    for i in range(N - 1):
        for j in range(N - 1):
            if i == j:
                D2[i][j] = (-1/3)*N*(N + 1)/(1 - x_[i]**2)/dr_dx_[i]**2
            else:
                D2[i][j] = -2/(x_[i] - x_[j])**2/(dr_dx[i]*dr_dx_[j])

    return D2


def compute_V_L(l, C):

    D2_ = D2[1: -1, 1: -1] - np.diag(l*(l + 1)/r**2)

    # boundary conditions:
    D2_N = D2[N][1: -1]
    V_N = (r/r_max)**2 * dr_dx[1: -1] * gll_weights
    B = -D2_N[:, np.newaxis] * V_N - np.diag((2*l + 1)/r)
    
    V_ = LA.inv(D2_) @ B
    
    V_L = np.sum(C**2 * V_ / r[:, np.newaxis], axis = 1)

    print(np.linalg.norm(V_*4*np.pi))
    print(np.max(np.abs(V_*4*np.pi)))
    print(f"||V_L||: {np.linalg.norm(V_L)}")

    return V_L


def direct_potential(l, C):

    W = np.zeros(N - 1)

    for L in range(l_cutoff):

        gaunt_coeff = float(gaunt(l, L, l, 0, 0, 0).n(64))

        V_L = compute_V_L(L, C)
        W += V_L * 4*np.pi/(2*L+1) * gaunt_coeff**2 # * 1/np.sqrt(4*np.pi)

    return W


def H_matrix(l):

    H = np.zeros((N - 1, N - 1))
    d = -2/r + l*(l + 1)/(2*r**2) # values on the diagonal
    np.fill_diagonal(H, d)
    H += -0.5*D2[1: -1, 1: -1]

    return H


def solve(l, C, H):

    V_dir = direct_potential(l, C)
    F = H + np.diag(V_dir)

    E, eigf = LA.eigh(F)

    f = eigf.T * P_x
    u = f / np.sqrt(dr_dx[1: -1])
    R = u / r
    R_norm = np.zeros((N - 1, N - 1))

    for i in range(len(f)):

        integral = np.sum(gll_weights * f[i]**2)
        R_norm[i] = R[i] / np.sqrt(integral)
        
    return(R_norm, u[0], E[0])



r_max = 30 # radial boundaries
N = 100
l_cutoff = 1
l = 0
x, r, dr_dx, P_x, gll_weights = setup_grid()
D1 = D1_matrix()
#D2 = D2_matrix()
D2 = D1 @ D1
H = H_matrix(l)



C = np.zeros(N - 1) # starting guess
R_norm, u_0, e_0_orb_update = solve(l, C, H)
print(e_0_orb_update)
diff = 1e-8 # dummy

while abs(diff) >= 1e-8:
    e_0_orb = e_0_orb_update
    C = u_0
    R_norm, u, e_0_orb_update = solve(l, C, H)
    diff = abs(e_0_orb - e_0_orb_update)
    print(e_0_orb_update)








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


