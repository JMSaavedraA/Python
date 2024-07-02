#!/usr/bin/env python3
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
plt.style.use('seaborn-talk')


"""
Tau leaping algorithm applied to the SEIR model with births and deaths
"""


"""
constants
"""

R0 = 1.5

sigma = 7.0/5.0
gamma = 7.0/7.0

mu = 7.0/(365.0*75.0)

N0 = 10**5

beta = R0**2*(gamma+mu)*(sigma+mu)/sigma

ti = 0.0 #initial time
tf = 60.0 #final time

X0 = ss.randint.rvs(0,10**1) # número aleatorio de latentemente infectados
Y0 = ss.randint.rvs(0,10**1) # número aleatorio de infectados
Z0 = 0                       # número inicial de recuperados 
W0 = N0-X0-Y0-Z0             # número inicial de susceptibles  

X = np.array((W0,X0,Y0,Z0))  # condiciones iniciales

tau = 1.0/7.0

"""
storage arrays
"""
Rate = np.zeros((8)) # propensidades

Change = np.zeros((8,4),dtype=int) # estequimetria

Change[0,:]=([-1,  1,  0,  0]) # contagio
Change[1,:]=([ 0, -1,  1,  0]) # infección
Change[2,:]=([ 0,  0, -1,  1]) # remoción
Change[3,:]=([ 1,  0,  0,  0]) # nacimiento de susceptible
Change[4,:]=([-1,  0,  0,  0]) # muerte de susceptible
Change[5,:]=([ 0, -1,  0,  0]) # muerte de latentemente infectado
Change[6,:]=([ 0,  0, -1,  0]) # muerte de infectado
Change[7,:]=([ 0,  0,  0, -1]) # muerte de recuperado

"""
tau leaping algorithm update
"""
def stoc_eqs(ti,X): 
    N = np.sum(X)
    Rate[0] = beta*X[0]*X[2]/N
    Rate[1] = sigma*X[1]    
    Rate[2] = gamma*X[2]
    Rate[3] = mu*N
    Rate[4] = mu*X[0]
    Rate[5] = mu*X[1] 
    Rate[6] = mu*X[2]  
    Rate[7] = mu*X[3] 
    for i in range(8):
        try:
            term = ss.poisson.rvs(Rate[i]*tau)
        except:
            term = 0
        X += Change[i,:]*term          
    return [ti+tau,X]

"""
tau leaping algorithm
"""
def Stoch_Iteration(ti,tf,X0):
    lop = 0
    T=[ti]
    S=[X0[0]]
    E=[X0[1]]    
    I=[X0[2]]
    R=[X0[3]]
    while T[lop] < tf:
        lop=lop+1
        [ti,X0] = stoc_eqs(ti,X0)
        T.append(ti)
        S.append(X0[0])
        E.append(X0[1])        
        I.append(X0[2])
        R.append(X0[3])        
    return np.array([T,S,E,I,R])

if __name__=="__main__":

    fig,ax = plt.subplots(4,1,sharex=True)
    T,S,E,I,R=Stoch_Iteration(ti,tf,X)
    
    ax[0].plot(T, S, 'g')
    ax[0].set_ylabel('Susceptible')
    ax[0].grid()    
    ax[1].plot(T, E, 'b')
    ax[1].set_ylabel ('Latente')   
    ax[1].grid()
    ax[2].plot(T, I, 'r')
    ax[2].set_ylabel ('Infectado')
    ax[2].grid()    
    ax[3].plot(T, R, 'k')
    ax[3].grid()    
    ax[3].set_xlabel ('Tiempo (días)')
    ax[3].set_ylabel ('Recuperado')