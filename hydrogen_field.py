import numpy as np
from scipy.special import sph_harm
from utils import setup_grid
from utils import D2_matrix
from utils import lebedev_parameters
from utils import V_regularized
from utils import setup_matrix
from numpy import linalg as LA



def V_time_averaged(alpha_0):

    n_int = 21 # integration points
    time_points = np.linspace(0, 1, n_int)
    dt = time_points[1] - time_points[0]
    V_t_averaged = V_regularized(R, r, theta, mu) # a_n = a_0
    
    for t in time_points[1: -1]:
        a = alpha_0 * np.sin(2 * np.pi * t)
        V_t_averaged += V_regularized(a + R, r, theta, mu)

    V_t_averaged *= dt

    return V_t_averaged


def solve_ground_state(alpha_0):

    V_t_averaged = V_time_averaged(alpha_0)
    v_l = 4*np.pi * np.sum(V_t_averaged[:, np.newaxis, :] * (Y * leb_weights.T)[np.newaxis, :, :], axis = 2)
    H = setup_matrix(N, v_l, r, D2, l_cutoff)
    E, eigf = LA.eigh(H)

    return E[0]


def energies():

    energies = np.zeros(4)
    alpha_0_array = np.array([0, 0.2, 0.8, 2])
    E_list = []

    for k in range(4):
        alpha_0 = alpha_0_array[k] 
        E = solve_ground_state(alpha_0)
        E_list.append(E)

    energies = np.array(E_list)

    dat = dict()
    dat["energies"] = energies
    dat["alpha"] = alpha_0_array
    np.savez('pes_hydrogen_atom.npz', **dat)


r_max = 30
N = 100
l_cutoff = 30
mu = 100 # regularization constant
R = 0

x, r, dr_dx, P_x, gll_weights = setup_grid(N, r_max)
D2 = D2_matrix(N, x, dr_dx)
theta, phi, leb_weights = lebedev_parameters(101)
Y = np.array([sph_harm(0, l, phi, theta).real for l in range(l_cutoff)])

E = solve_ground_state(alpha_0 = 0)
print(E)


