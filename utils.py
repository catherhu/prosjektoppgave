import numpy as np
from scipy.special import erf
from sympy.physics.wigner import gaunt


def setup_grid(N, r_max):

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
    gll_weights = 2/(N*(N + 1)*P_x**2)

    return(x[1: -1], r[1: -1], dr_dx[1: -1], P_x[1: -1], gll_weights[1: -1])


def D2_matrix(N, x, dr_dx):

    D = np.zeros((N - 1, N - 1))

    for i in range(N - 1):

        for j in range(N - 1):

            if i == j:
                D[i][j] = (-1/3)*N*(N + 1)/(1 - x[i]**2)/dr_dx[i]**2
            else:
                D[i][j] = -2/(x[i] - x[j])**2/(dr_dx[i]*dr_dx[j])

    return D


def lebedev_parameters(N_lebedev):

    coord = np.loadtxt("Lebedev/lebedev_%03d.txt" % N_lebedev)
    phi = coord[:, 0] * np.pi / 180 + np.pi
    theta = coord[:, 1] * np.pi / 180
    leb_weights = coord[:, 2]

    return(theta, phi, leb_weights)


def V_regularized(a, r, theta, mu):
    r_ = r[:, np.newaxis]
    theta_ = theta[np.newaxis, :]
    r_a = np.sqrt(r_**2 - 2*a*r_*np.cos(theta_) + a**2)
    V_reg = np.where(r_a == 0, 2*mu/np.sqrt(np.pi), erf(mu*r_a)/r_a)

    return V_reg


def setup_submatrix(N, l1, l2, V, r, D2):

    H = np.zeros((N - 1, N - 1))
    d = -V 

    if l1 == l2:
        H += -0.5*D2
        d += l1 * (l1 + 1) / (2 * r ** 2)

    np.fill_diagonal(H, H.diagonal() + d)
     
    return H


def setup_matrix(N, v_l, r, D2, l_cutoff):

    size = (N - 1) * l_cutoff
    H_block = np.zeros((size, size))
    
    for l1 in range(l_cutoff):

        for l2 in range(l1, l_cutoff):
            V = np.zeros(N - 1)

            for l3 in range(l_cutoff):
                gaunt_coeff = float(gaunt(l1, l2, l3, 0, 0, 0).n(64))
                V += v_l[:, l3] * gaunt_coeff

            H = setup_submatrix(N, l1, l2, V, r, D2)
            H_block[l1*(N - 1): (l1 + 1)*(N - 1), l2*(N - 1): (l2 + 1)*(N - 1)] = H 
            H_block[l2*(N - 1): (l2 + 1)*(N - 1), l1*(N - 1): (l1 + 1)*(N - 1)] = H

    return H_block


def radial_component(eigf, N, l_cutoff, r, dr_dx, P_x, gll_weights):

    split_eigf = np.hsplit(eigf.T, l_cutoff)
    sum_l_eigf = np.sum(split_eigf, axis = 0)
    f = sum_l_eigf * P_x
    u = f / np.sqrt(dr_dx)
    R = u / r
    R_norm = np.zeros_like(R)

    for i in range((N - 1)*l_cutoff):
        integral = np.sum(gll_weights * f[i]**2)
        R_norm[i] = R[i] / np.sqrt(integral)

    return(R_norm)