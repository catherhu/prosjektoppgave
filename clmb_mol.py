import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA


N_steps = 1000 # number of steps
L = 10 # x boundaries
h = L/N_steps
x = np.linspace(-L, L, N_steps + 1) # positions

alpha = 1
R = 3
V = -1/np.sqrt((x - R)**2 + alpha) - 1/np.sqrt((x + R)**2 + alpha) # potential

d = 1/(h**2) + V # main diagonal
e = np.full(len(d)-1, -1/(2*h**2)) # sub and super diagonal

# set up matrix:
size = len(d)
M = np.zeros((size, size))
np.fill_diagonal(M, d)
np.fill_diagonal(M[1:], e)
np.fill_diagonal(M[:,1:], e)

# solve:
E, psi = LA.eigh(M)

# plot:
plt.figure()
plt.grid()
plt.plot(x, V)
for i in range(4):
    shift = E[i]
    psi_i = psi[:,i] / np.sqrt(np.trapz(psi[:,i]**2, x))
    plt.plot(x, psi_i**2 + shift)
plt.savefig("clmb_mol.png")
