import numpy as np
from math import factorial
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.special import sph_harm
from scipy.interpolate import CubicSpline
from sympy.physics.wigner import gaunt
import plotly.graph_objects as go
import os

# constants:
r_max = 30 # radial boundaries
L = 1 # mapping constant
alpha = 2*L/r_max # mapping constant

a = 1 # spacial shift of potential

N = 100
l_cutoff = 20


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


def potential(r, l1, l2, l_cutoff, a):
    
    a_array = np.ones(N - 1) * a
    V1 = np.zeros(N - 1)
    V2 = np.zeros(N - 1)

    for l3 in range(l_cutoff):
        gaunt_coeff = float(gaunt(l1, l2, l3, 0, 0, 0).n(64))
        V1 += np.sqrt(4*np.pi/(2*l3 + 1)) * np.minimum(a_array, r[1: -1]) ** l3 / np.maximum(a_array, r[1: -1]) ** (l3 + 1) * gaunt_coeff
        V2 += (-1)**l3 * np.sqrt(4*np.pi/(2*l3 + 1)) * np.minimum(-a_array, r[1: -1]) ** l3 / np.maximum(a_array, r[1: -1]) ** (l3 + 1) * gaunt_coeff
        V = V1 + V2
    return -V


def setup_submatrix(l1, l2, l_cutoff, N, x, r, dr_dx):

    V = potential(r, l1, l2, l_cutoff, a)

    M = np.zeros((N + 1, N + 1))
    d = np.zeros(N + 1)
    d[1: -1] = V 

    if l1 == l2:
        d[1: -1] += l1 * (l1 + 1) / (2 * r[1: -1] ** 2)

    np.fill_diagonal(M, d)

    if l1 == l2:
        D = np.zeros((N + 1, N + 1))

        for i in range(1, N):

            for j in range(1, N):
                if i == j:
                    D[i][j] = (-1/3)*N*(N + 1)/(1 - x[i]**2)/dr_dx[i]**2
                else:
                    D[i][j] = -2/(x[i] - x[j])**2/(dr_dx[i]*dr_dx[j])

        M += -0.5*D

    return M[1: -1, 1: -1]


def setup_matrix(l_cutoff, N, x, r, dr_dx):

    size = (N - 1) * l_cutoff
    M_block = np.zeros((size, size))

    for l1 in range(l_cutoff):

        for l2 in range(l_cutoff):
            M = setup_submatrix(l1, l2, l_cutoff, N, x, r, dr_dx)

            """
            for row in M:
                print("    ".join(f"{element:5.2f}" for element in row))
            """

            for i in range(N - 1):

                for j in range(N - 1):
                    i_ = i + l1 * (N - 1)
                    j_ = j + l2 * (N - 1)
            
                    M_block[i_][j_] = M[i][j]

    """
    plt.figure()
    plt.imshow(M_block)
    plt.savefig("M_matrix.png")   
    for row in M_block:
        print("    ".join(f"{element:5.2f}" for element in row))
    """

    return M_block


def radial(l_cutoff, N):

    x, r, dr_dx, P_x = setup_grid(N)
    M = setup_matrix(l_cutoff, N, x, r, dr_dx)

    E, eigf = LA.eigh(M)
    print(E[0:4])
    f = np.hsplit(eigf.T, l_cutoff) * P_x[1: -1]
    u = f / np.sqrt(dr_dx[1: -1])
    
    R = u / r[1: -1]

    R_norm = np.zeros_like(R)

    for i in range(l_cutoff):

        for j in range((N - 1)*l_cutoff):
            integral = np.sum(2 / (N * (N + 1) * P_x[1: -1] ** 2) * f[i][j]**2)
            R_norm[i][j] = R[i][j] / np.sqrt(integral)
        
    return(r[1: -1], R_norm)


def plot_pdf(l_cutoff, N, n_cutoff):

    x_mesh, y_mesh, z_mesh = np.mgrid[-17 : 17 : (N - 1)*1j, -17 : 17 : (N - 1)*1j, -17 : 17 : (N - 1)*1j]

    r_int = np.sqrt(x_mesh**2 + y_mesh**2 + z_mesh**2)
    phi = np.arctan2(np.sqrt(x_mesh**2 + y_mesh**2), z_mesh)
    theta = np.arctan2(y_mesh, x_mesh)

    r, R = radial(l_cutoff, N)

    R_l_sum = np.sum(R, axis = 0)

    for l in [0, 1, 2]:
        Y = sph_harm(0, l, theta, phi)

        for i in range(l, n_cutoff):
                cs = CubicSpline(r, R_l_sum[i])
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
                
                fig.write_image(os.path.join("plots_hydrogen2+", f"hydrogen2+_pdf_{i + 1}{l}{0}.png"))
                fig.write_html(os.path.join("htmls_hydrogen2+", f"hydrogen2+_pdf_{i + 1}{l}{0}.html"))

radial(l_cutoff, N)
#plot_pdf(l_cutoff, N, n_cutoff = 3)
