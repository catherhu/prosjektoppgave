import numpy as np
from numpy import linalg as LA
from scipy.special import sph_harm
from utils import setup_grid
from utils import D2_matrix
from utils import lebedev_parameters
from utils import V_regularized
from utils import setup_matrix
from plotting_scripts import plot_potential_energy_surfaces
from plotting_scripts import plot_exact_vs_regularized_potential


def V_time_averaged(alpha_0, internuclear_distance):

    R1 = internuclear_distance/2
    R2 = -internuclear_distance/2

    n_int = 21 # integration points
    time_points = np.linspace(0, 1, n_int)
    dt = time_points[1] - time_points[0]
    V_t_averaged = V_regularized(R1, r, theta, mu) + V_regularized(R2, r, theta, mu) # a_n = a_0
    
    for t in time_points[1: -1]:
        a = alpha_0 * np.sin(2 * np.pi * t)
        V_t_averaged += V_regularized(a + R1, r, theta, mu) + V_regularized(a + R2, r, theta, mu)

    V_t_averaged *= dt

    return V_t_averaged


def solve_ground_state(alpha_0, internuclear_distance):

    V_t_averaged = V_time_averaged(alpha_0, internuclear_distance)
    v_l = 4*np.pi * np.sum(V_t_averaged[:, np.newaxis, :] * (Y * leb_weights.T)[np.newaxis, :, :], axis = 2)
    H = setup_matrix(N, v_l, r, D2, l_cutoff)
    E, eigf = LA.eigh(H)

    return E[0]


def potential_energy_surfaces():

    energies = np.zeros((4, 20))
    alpha_0_array = np.array([0, 0.2, 0.8, 2])
    internuc_dist_array = np.linspace(1, 5, 20)

    n_iteration = 0

    for k in range(4):
        alpha_0 = alpha_0_array[k] 
        E_list = []
    
        for internuclear_distance in internuc_dist_array:
            E = solve_ground_state(alpha_0, internuclear_distance) + 1/internuclear_distance
            E_list.append(E)
            n_iteration += 1
            print(f"iteration {n_iteration} complete")
        energies[k] = np.array(E_list)

    dat = dict()
    dat["energies"] = energies
    dat["internuclear_distance"] = internuc_dist_array
    dat["alpha_0"] = alpha_0_array
    np.savez("pes_dihydrogen_cation.npz", **dat)


r_max = 30
N = 200
l_cutoff = 30
mu = 30 # regularization constant

x, r, dr_dx, P_x, gll_weights = setup_grid(N, r_max)
D2 = D2_matrix(N, x, dr_dx)
theta, phi, leb_weights = lebedev_parameters(101)
Y = np.array([sph_harm(0, l, phi, theta).real for l in range(l_cutoff)])

plot_exact_vs_regularized_potential(r, Y, theta, leb_weights)
#potential_energy_surfaces()
#plot_potential_energy_surfaces()
#E = solve_ground_state(alpha_0 = 0, internuclear_distance = 2) + 1/2
#print(E)


















