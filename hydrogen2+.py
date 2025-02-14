import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.special import sph_harm
from scipy.special import erf
from scipy.interpolate import CubicSpline
from sympy.physics.wigner import gaunt
import plotly.graph_objects as go
import os

# constants:
r_max = 30 # radial boundaries
L = 1 # mapping constant
alpha = 2*L/r_max # mapping constant
mu = 5 # regularization constant

a = 2 # spacial shift of potential

N = 100 # grid size
l_cutoff = 10


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
    #r = L*(1 + x)/(1 - x + alpha) # radial grid points
    dr_dx = r_max/2 * np.ones_like(r)
    #dr_dx = L*(2 + alpha)/(1 - x + alpha)**2

    return(x, r, dr_dx, P_x)


def erf_func(r, a, theta, mu):

    r_a = np.sqrt(r**2 - 2*a*r*np.cos(theta) + a**2)

    if r_a == 0:
        return 2*mu/np.sqrt(np.pi)
    else:
        return erf(mu*r_a)/r_a


def test_potential_regularization(a, mu, N):

    x, r, dr_dx, P_x = setup_grid(N)
    r_inner = r[1:-1]
    print(r_inner)
    a_array = np.ones(N - 1) * a
    N_lebedev = 101
    coord = np.loadtxt("Lebedev/lebedev_%03d.txt" % N_lebedev)
    phi = coord[:, 0] * np.pi / 180 + np.pi
    theta = coord[:, 1] * np.pi / 180
    lebedev_weights = coord[:, 2]
    n_theta = len(theta)

    print(f"a: {a}, mu: {mu}, N: {N}")

    erf_array = np.zeros((N-1, n_theta))

    for i in range(N-1):
        for j in range(n_theta):
            erf_array[i,j] = erf_func(r_inner[i], a, theta[j], mu)
    print(erf_array[0,0])
    #plt.plot(r_inner, erf_array[:, 0])
    #plt.plot(r_inner, erf_array[:, n_theta//2], label = "n_theta/2")
    #plt.legend()
    #plt.savefig("test.png")
    for l in range(3):

        V_l = np.sqrt(4*np.pi/(2*l + 1)) * np.minimum(a_array, r[1: -1]) ** l / np.maximum(a_array, r[1: -1]) ** (l + 1)

        

        Y = sph_harm(0, l, phi, theta).real

        f_l = np.zeros(N - 1)
        for j in range(n_theta):
            f_l += 4*np.pi*erf_array[:,j]*Y[j]*lebedev_weights[j]
        """
        for i in range(N - 1):
            for j in range(n_theta):
                f_l[i] += 4 * np.pi * erf_func(r_inner[i], a, theta[j], mu) * Y[j] * lebedev_weights[j]
        """
        #f_l *= np.sqrt((2*l + 1)/(4*np.pi))
        
        plt.plot(r[1: -1], V_l, label = "V_l")
        plt.plot(r[1: -1], f_l, label = r"$f_%d$" % l)
        plt.legend()
        plt.savefig("test_potential_regularization.png")

def potential(r, l1, l2, l_cutoff, a):
    
    a_array = np.ones(N - 1) * a
    V1 = np.zeros(N - 1)
    V2 = np.zeros(N - 1)

    for l3 in range(l_cutoff):
        gaunt_coeff = float(gaunt(l1, l2, l3, 0, 0, 0).n(64))
    
        V1 += np.sqrt(4*np.pi/(2*l3 + 1)) * np.minimum(a_array, r[1: -1]) ** l3 / np.maximum(a_array, r[1: -1]) ** (l3 + 1) * gaunt_coeff
        #V2 += (-1)**l3 * np.sqrt(4*np.pi/(2*l3 + 1)) * np.minimum(a_array, r[1: -1]) ** l3 / np.maximum(a_array, r[1: -1]) ** (l3 + 1) * gaunt_coeff
  
    V = V1 + V2

    return -V


def setup_submatrix(l1, l2, l_cutoff, N, x, r, dr_dx):

    V = potential(r, l1, l2, l_cutoff, a)

    M = np.zeros((N + 1, N + 1))
    d = np.zeros(N + 1)
    d[1: -1] = V 

    if l1 == l2:
        d[1: -1] += l1 * (l1 + 1) / (2 * r[1: -1] ** 2) - 1/r[1: -1]

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

            for i in range(N - 1):

                for j in range(N - 1):
                    i_ = i + l1 * (N - 1)
                    j_ = j + l2 * (N - 1)
            
                    M_block[i_][j_] = M[i][j]

    return M_block


def radial(l_cutoff, N):

    x, r, dr_dx, P_x = setup_grid(N)
    M = setup_matrix(l_cutoff, N, x, r, dr_dx)

    E, eigf = LA.eigh(M)

    print(E[0] + 0.5)
    E_true = -0.597
    error = abs(E_true - E[0])

    split_eigf = np.hsplit(eigf.T, l_cutoff)
    sum_l_eigf = np.sum(split_eigf, axis = 0)
    f = sum_l_eigf * P_x[1: -1]
    u = f / np.sqrt(dr_dx[1: -1])
    
    R = u / r[1: -1]
    R_norm = np.zeros_like(R)

    for i in range((N - 1)*l_cutoff):
        integral = np.sum(2 / (N * (N + 1) * P_x[1: -1] ** 2) * f[i]**2)
        R_norm[i] = R[i] / np.sqrt(integral)
        
    return(r[1: -1], R_norm, error)


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

test_potential_regularization(a, mu, N)
#plot_pdf(l_cutoff, N, n_cutoff = 3)
