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
x[1: -1] = np.polynomial.legendre.legroots(dP_dx) # polynomial derivative roots
P_x = np.polynomial.legendre.legval(x[1: -1], P) # evaluating polynomial at grid points

r = L*(1 + x[1: -1])/(1 - x[1: -1] + alpha) # radial grid points

# potential:
V = -1/r

# set up matrix:
M = np.zeros((N + 1, N + 1))
d = np.zeros(N + 1)
d[1: -1] = V + l*(l + 1)/(2*r) # values on the diagonal
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

# solve:
E, eigf = LA.eigh(M[1: -1, 1: -1])

f = eigf.T * P_x
u = f / np.sqrt(dr_dx[1: -1])
R = u / r
Y = 1 # add later
psi = R * Y

# plot radial function:
plt.figure(1)
plt.grid()
for i in range (3):
    R_i = R[i]
    plt.plot(r, -R_i, label = f"{i + 1}{l}")
plt.legend()
plt.savefig("3d_hydrogen_radial.png")

# contour plot:
plt.figure(2)
phi = np.linspace(1, 2*np.pi, N - 1)
x_points = r * np.cos(phi)
y_points = r * np.sin(phi)
x_mesh, y_mesh = np.meshgrid(x_points, y_points)
psi_0 = psi[0]/np.sqrt(x_mesh**2 + y_mesh**2)
plt.contour(x_mesh, y_mesh, psi_0**2, levels=10, cmap='viridis')
plt.colorbar()
plt.savefig("3d_hydrogen_contour.png")

#print(E)





    
