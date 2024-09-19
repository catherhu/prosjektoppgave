import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy import integrate

# grid:
N_steps = 1000 # number of steps
L = 20 # x boundaries
h = L/N_steps # step size
x = np.linspace(-L, L, N_steps + 1) # positions

# constants:
alpha = 1 # regularization constant
a_0 = 10
T = 2*np.pi # period

# potential:
t = np.linspace(0, T, N_steps + 1) # time
V = np.zeros(N_steps + 1)
for i in range(len(x)):
    V[i] = -1/T * integrate.trapezoid(2 / np.sqrt((x[i] + a_0 * np.cos(t)) ** 2 + alpha), x = t)

# set up matrix:
d = 1/(h**2) + V # main diagonal
e = np.full(len(d)-1, -1/(2*h**2)) # sub and super diagonal
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
plt.plot(x, V, label = "V")
for i in range(4):
    shift = E[i]
    psi_i = psi[:,i] / np.sqrt(np.trapz(psi[:,i]**2, x))
    plt.plot(x, psi_i**2 + shift)
plt.legend()
plt.savefig("1d_field_1e_1n.png")