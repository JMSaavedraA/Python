#!/usr/bin/env python3
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from matplotlib import cm

"""
Solución de la ecuacion maestra de la química para el modelo SIS.
Ejemplo tomado de la sección 2 del artículo: 

Keeling, Matthew James, and Joshua V. Ross. "On methods for studying 
stochastic disease dynamics." Journal of the Royal Society Interface 5.19 
(2008): 171-181.
"""


N = 100 # tamaño de la población
g = 1 # tasa de recuperación
beta = 2.0/N # tasa de contacto
epsilon = 0.01 # tasa de importación de infecciones

"""
Forma ma matriz Q
"""

a = np.arange(N)
d = -(beta*(N-a)*(a+epsilon) + a*g)
dl = (a+1)*g
du = beta*(N-a)*(epsilon+a)

Q = np.diag(d,0)+np.diag(du[:-1],-1)+np.diag(dl[:-1],1)

P0 = np.zeros(N)
P0[2] = 1

t = np.linspace(0,100.0,100)

soln = np.zeros((N,N))

for index in np.arange(100):
	soln[:,index] = expm(t[index]*Q)@P0

x,y = np.meshgrid(a,t)

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1, projection='3d')
X, Y = np.meshgrid(t, a)
surf = ax.plot_surface(X, Y, soln, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.set_zlim3d(-1.01, 1.01)
fig.colorbar(surf, shrink=0.5, aspect=5)
