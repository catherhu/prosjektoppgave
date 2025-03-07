import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.special import sph_harm
from scipy.special import erf
from scipy.interpolate import CubicSpline
from sympy.physics.wigner import gaunt
import plotly.graph_objects as go
import time 
import os


r_max = 30 # radial boundaries
N = 50 # grid size
l_cutoff = 10
s = 2 # internuclear distance
alpha_0 = 0.2
mu = 30 # regularization constant


def setup_grid(N):

    P = np.zeros(N + 1) # polynomial coefficients
    P[-1] = 1 
    dP_dx = np.polynomial.legendre.legder(P) # derivative coefficients
    x = np.zeros(N + 1) # grid points
    x[0] = -1 
    x[N] = 1
    x[1: -1] = np.polynomial.legendre.legroots(dP_dx) # polynomial derivative roots
    P_x = np.polynomial.legendre.legval(x, P) # evaluating polynomial at grid points
    r = r_max/2 * (x + 1)
    dr_dx = r_max/2 * np.ones_like(r)
    
    return(x, r[1: -1], dr_dx, P_x)


def erf_func(r, a, theta, mu):

    r_a = np.sqrt(r**2 - 2*a*r*np.cos(theta) + a**2)
    erf_array = np.where(r_a == 0, 2*mu/np.sqrt(np.pi), erf(mu*r_a)/r_a)

    return erf_array


def lebedev(N_lebedev):

    coord = np.loadtxt("Lebedev/lebedev_%03d.txt" % N_lebedev)
    phi = coord[:, 0] * np.pi / 180 + np.pi
    theta = coord[:, 1] * np.pi / 180
    leb_weights = coord[:, 2]

    return(theta, phi, leb_weights)


def potential(l1, l2, l_cutoff, r, a, s, theta, leb_weights, Y):
    
    V1 = np.zeros(N - 1)
    V2 = np.zeros(N - 1)

    erf_array_1 = erf_func(r[:, np.newaxis], a, theta[np.newaxis, :], mu)
    erf_array_2 = erf_func(r[:, np.newaxis], a - s, theta[np.newaxis, :], mu)

    for l3 in range(l_cutoff):
        gaunt_coeff = float(gaunt(l1, l2, l3, 0, 0, 0).n(64))  
        f_l_1 = np.sum(4 * np.pi * erf_array_1 * Y[l3] * leb_weights, axis = 1)
        f_l_2 = np.sum(4 * np.pi * erf_array_2 * Y[l3] * leb_weights, axis = 1)
        V1 += f_l_1 * gaunt_coeff
        V2 += f_l_2 * gaunt_coeff
        
    V = V1 + V2

    return -V


def field_potential(l1, l2, l_cutoff, r, s, alpha_0, theta, leb_weights, Y):

    n_int = 20 # integration points
    n_array = np.linspace(0, n_int, n_int + 1)
    a_0 = 1 

    V_field = potential(l1, l2, l_cutoff, r, a_0, s, theta, leb_weights, Y) # a_n = a_0
    
    for n in n_array[1: -1]:
        a = a_0 + alpha_0 * np.sin(np.pi * n / n_int)
        V = potential(l1, l2, l_cutoff, r, a, s, theta, leb_weights, Y)
        V_field += V
    
    return V_field / n_int


def D_matrix(N, x, dr_dx):

    D = np.zeros((N + 1, N + 1))

    for i in range(1, N):

        for j in range(1, N):

            if i == j:
                D[i][j] = (-1/3)*N*(N + 1)/(1 - x[i]**2)/dr_dx[i]**2
            else:
                D[i][j] = -2/(x[i] - x[j])**2/(dr_dx[i]*dr_dx[j])

    return D


def setup_submatrix(l1, l2, l_cutoff, N, r, s, alpha_0, theta, leb_weights, Y, D):

    V = field_potential(l1, l2, l_cutoff, r, s, alpha_0, theta, leb_weights, Y)

    M = np.zeros((N + 1, N + 1))
    d = np.zeros(N + 1)
    d[1: -1] = V 

    if l1 == l2:
        M += -0.5*D
        d[1: -1] += l1 * (l1 + 1) / (2 * r ** 2)

    np.fill_diagonal(M, M.diagonal() + d)
        
    return M[1: -1, 1: -1]


def setup_matrix(l_cutoff, N, r, s, alpha_0, theta, leb_weights, Y, D):

    size = (N - 1) * l_cutoff
    M_block = np.zeros((size, size))
    
    for l1 in range(l_cutoff):

        for l2 in range(l_cutoff):
            M = setup_submatrix(l1, l2, l_cutoff, N, r, s, alpha_0, theta, leb_weights, Y, D)
            M_block[l1*(N - 1): (l1 + 1)*(N - 1), l2*(N - 1): (l2 + 1)*(N - 1)] = M

    return M_block


def radial(l_cutoff, N, r, s, alpha_0, theta, leb_weights, Y, D, dr_dx, P_x):

    M = setup_matrix(l_cutoff, N, r, s, alpha_0, theta, leb_weights, Y, D)
   
    E, eigf = LA.eigh(M)

    split_eigf = np.hsplit(eigf.T, l_cutoff)
    sum_l_eigf = np.sum(split_eigf, axis = 0)
    f = sum_l_eigf * P_x[1: -1]
    u = f / np.sqrt(dr_dx[1: -1])
    R = u / r
    R_norm = np.zeros_like(R)

    for i in range((N - 1)*l_cutoff):
        integral = np.sum(2 / (N * (N + 1) * P_x[1: -1] ** 2) * f[i]**2)
        R_norm[i] = R[i] / np.sqrt(integral)

    W = 1/s

    return(E[0] + W, R_norm)





x, r, dr_dx, P_x = setup_grid(N)
D = D_matrix(N, x, dr_dx)
theta, phi, leb_weights = lebedev(101)
Y = np.array([sph_harm(0, l, phi, theta).real for l in range(l_cutoff)])

s = 2
E, R = radial(l_cutoff, N, r, s, alpha_0, theta, leb_weights, Y, D, dr_dx, P_x)
print(E)


"""
s_array = np.linspace(0, 6, 20)   
E_values = []

for s in s_array[1:]:
    E, R = radial(l_cutoff, N, r, s, alpha_0, theta, leb_weights, Y, dr_dx, P_x)
    E_values.append(E)

plt.figure()
plt.plot(s_array[1:], E_values)
plt.savefig("energy.png")
"""


"""
def plot_pdf(l_cutoff, N, n_cutoff):

    x_mesh, y_mesh, z_mesh = np.mgrid[-17 : 17 : (N - 1)*1j, -17 : 17 : (N - 1)*1j, -17 : 17 : (N - 1)*1j]

    r_int = np.sqrt(x_mesh**2 + y_mesh**2 + z_mesh**2)
    phi = np.arctan2(np.sqrt(x_mesh**2 + y_mesh**2), z_mesh)
    theta = np.arctan2(y_mesh, x_mesh)

    r, R = radial(l_cutoff, N)

    for l in [0, 1, 2]:
        Y = sph_harm(0, l, theta, phi)

        for i in range(l, n_cutoff):
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
                
                fig.write_image(os.path.join("plots_hydrogen2+", f"hydrogen2+_pdf_{i + 1}{l}{0}.png"))
                fig.write_html(os.path.join("htmls_hydrogen2+", f"hydrogen2+_pdf_{i + 1}{l}{0}.html"))
"""

