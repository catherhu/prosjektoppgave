import matplotlib.pyplot as plt
import numpy as np
from utils import V_regularized
from scipy.interpolate import CubicSpline


def plot_exact_vs_regularized_potential(r, Y, theta, leb_weights):

    R = 1

    plt.figure(1)
    l = 0
    for mu in (5, 10, 30):
        V_reg = V_regularized(R, r, theta, mu)
        v_reg = 4*np.pi * np.sum(V_reg[:, np.newaxis, :] * (Y * leb_weights.T)[np.newaxis, :, :], axis = 2)
        v_l_reg = v_reg[:, l]
        v_l = np.sqrt(4*np.pi/(2*l + 1)) * np.minimum(r, R) ** l / np.maximum(r, R) ** (l + 1)
        plt.plot(r, v_l_reg, label=rf"$v_l^{{\text{{reg}}}},\ \mu = {mu}$")
    plt.plot(r, v_l, linestyle = "--", label = r"$v_l$")
    plt.legend()
    plt.xlabel(r"$r$")
    plt.xlim(0, 2)
    plt.ylim(3.2, 3.6)
    plt.savefig(f"regularized_potential_l_{l}.png")

    plt.figure(2)
    l = 1
    for mu in (5, 10, 30):
        V_reg = V_regularized(R, r, theta, mu)
        v_reg = 4*np.pi * np.sum(V_reg[:, np.newaxis, :] * (Y * leb_weights.T)[np.newaxis, :, :], axis = 2)
        v_l_reg = v_reg[:, l]
        v_l = np.sqrt(4*np.pi/(2*l + 1)) * np.minimum(r, R) ** l / np.maximum(r, R) ** (l + 1)
        plt.plot(r, v_l_reg, label=rf"$v_l^{{\text{{reg}}}},\ \mu = {mu}$")
    plt.plot(r, v_l, linestyle = "--", label = r"$v_l$")
    plt.legend()
    plt.xlabel(r"$r$")
    plt.xlim(0, 2)
    plt.ylim(1.25, 2.1)
    plt.savefig(f"regularized_potential_l_{l}.png")

    plt.figure(3)
    mu = 30
    for l in range(3):
        V_reg = V_regularized(R, r, theta, mu)
        v_reg = 4*np.pi * np.sum(V_reg[:, np.newaxis, :] * (Y * leb_weights.T)[np.newaxis, :, :], axis = 2)
        v_l_reg = v_reg[:, l]
        v_l = np.sqrt(4*np.pi/(2*l + 1)) * np.minimum(r, R) ** l / np.maximum(r, R) ** (l + 1)
        plt.plot(r, v_l_reg, label=rf"$v_l^{{\text{{reg}}}},\ l = {l}$")
        plt.plot(r, v_l, linestyle = "--", label = rf"$v_l, \ l = {l}$")
    plt.legend()
    plt.xlabel(r"$r$")
    plt.savefig(f"regularized_potential.png")


def plot_potential_energy_surfaces():

    data = np.load("pes_dihydrogen_cation.npz")
    energies_dihydrogen_cation = data["energies"]
    internuclear_distance = data["internuclear_distance"]
    alpha_0 = data["alpha_0"]

    data = np.load("pes_hydrogen_atom.npz")
    energies_hydrogen_atom = data["energies"]

    plt.figure(1)

    fine_grid = np.linspace(1, 5, 1000)

    for k in range(4):
        plt.scatter(internuclear_distance, energies_dihydrogen_cation[k], s = 5, label = rf"$\alpha_0 = {alpha_0[k]}$")
        cs = CubicSpline(internuclear_distance, energies_dihydrogen_cation[k])
        energies_interpolated = cs(fine_grid)
        plt.plot(fine_grid, energies_interpolated)

    plt.xlabel("Internuclear Distance (a. u.)")
    plt.ylabel("Potential Energy (a. u.)")
    plt.legend()   
    plt.savefig("pes_dihydrogen_cation.png") 

    plt.figure(2)

    for k in range(4):
        stabilization_energy = energies_dihydrogen_cation[k] - energies_hydrogen_atom[k]
        plt.scatter(internuclear_distance, stabilization_energy, s = 5, label = rf"$\alpha_0 = {alpha_0[k]}$")
        cs = CubicSpline(internuclear_distance, stabilization_energy)
        energies_interpolated = cs(fine_grid)
        plt.plot(fine_grid, energies_interpolated)

    plt.xlabel("Internuclear Distance (a. u.)")
    plt.ylabel("Potential Energy (a. u.)")
    plt.legend()   
    plt.savefig("stabilization_energy.png") 