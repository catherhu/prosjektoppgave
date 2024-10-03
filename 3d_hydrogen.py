import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy import integrate

# constants:
r_max = 10 # radial boundaries
l = 0 # angular momentum
L = 1 # mapping constant
alpha = 2*L/r_max # mapping constant

# grid:
N = 100 # polynomial degree
P = np.zeros(N + 1) # polynomial coefficients
P[-1] = 1 
dP_dx = np.polynomial.legendre.legder(P) # derivative coefficients

x = np.zeros(N + 1) # grid points
x[0] = -1 
x[N] = 1
x [1: -1] = np.polynomial.legendre.legroots(dP_dx) # polynomial derivative roots
P_x = np.polynomial.legendre.legval(x[1: -1], P) # evaluating polynomial at grid points

r = L*(1 + x)/(1 - x + alpha) # radial grid points

# potential:
V = np.zeros(N + 1)
V[1: -1] = -1/r[1: -1] 

# set up matrix:

M = np.zeros((N + 1, N + 1))
d = V + l*(l + 1)/(2*r) # values on the diagonal
np.fill_diagonal(M, d)

dr_dx = L*(2 + alpha)/(1 - x + alpha)**2
D = np.zeros((N + 1, N + 1))

for i in range(1, N):
    for j in range(1, N):
        if i == j:
            D[i][j] = (-1/3)*N*(N + 1)/(1 - x[i]**2)/dr_dx[i]**2
        else:
            D[i][j] = -2/(x[i] - x[j])**2/(dr_dx[i]*dr_dx[j])

M += -0.5*D
M = M[1: -1, 1: -1]

# solve:
E, eigf = LA.eigh(M)
f = eigf.T * P_x
"""
go back to original function psi
make 1D plots
make contour plots
"""

# plot:
phi = np.linspace(1, 2*np.pi, N - 1)
x_points = r[1: -1]*np.cos(phi)
y_points = r[1: -1]*np.sin(phi)
X, Y = np.meshgrid(x_points, y_points)
psi_0 = psi[0]/np.sqrt(X**2 + Y**2)
plt.contour(X, Y, psi_0**2, levels=10, cmap='viridis')
plt.colorbar()
plt.savefig("3d_hydrogen.png")

print(E)





    
