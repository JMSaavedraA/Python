#!/usr/bin/env python3
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
plt.style.use('seaborn-talk')

"""
Simulación de una infección viral. Los parámetros corresponden a una infección causada por VIH.

NOTA: sample_size es sl número de simulaciones. Modificala para ver el periodo eclipse al inicio 
de la infección.
"""

ND = 10 #Tiempo máximo
alfa1=30.6;
alfa2=25.6;
gamma=4.8;
beta=6.8;



u0 = 1
v0 = 1
sample_size = 100

Change=np.zeros((4,2),dtype=int) # stoichiometry
Change[0,:]=([1, 0]);
Change[1,:]=([-1, 0]);
Change[2,:]=([0, 1]);
Change[3,:]=([0, -1]);

def stoc_eqs(T,X):
    Rate=np.zeros((4))
    Rate[0] = alfa1/(1+X[1]**beta) #target cell production
    Rate[1] = X[0] #infection
    Rate[2] = alfa2/(1+X[0]**gamma) #lysis
    Rate[3] = X[1] #budding                   
    R1=ss.uniform.rvs()
    R2=ss.uniform.rvs()
    ts = -np.log(R2)/(np.sum(Rate))
    j = np.where(np.cumsum(Rate)>=R1*np.sum(Rate))[0][0]
    X[range(2)]+=Change[j,:]    
    return [T+ts,X]


def Stoch_Iteration(T0,X0):
    lop = 0
    T = [T0]
    U = [X0[0]]
    V = [X0[1]]
    while T[lop] < ND:        
        [T0,X0] = stoc_eqs(T0,X0)
        T.append(T0)
        U.append(X0[0])
        V.append(X0[1])
        lop += 1        
    return [T,U,V]


Ufin=[]
Vfin=[]
#fig,ax = plt.subplots(1,1)
for k in np.arange(sample_size):
    X0 = np.array([u0,v0])
    [T,U,V]=Stoch_Iteration(0,X0)    
    Ufin.append(U[-1])
    Vfin.append(V[-1])
    #ax.plot(T, U, 'r', lw=0.5, alpha=0.5)

plt.hist(Ufin)
plt.show()
#ax.set_xlabel ('Tiempo (Días)')
#ax.set_ylabel ('Virus')
