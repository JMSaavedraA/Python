#!/usr/bin/env python3
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
plt.style.use('seaborn-talk')


"""
Gillespie algorithm applied to the SIR epidemic model with births and deaths
"""


"""
constants
"""
R0 = 5.0 # set R0
gamma = 1.0/7.0 # set the recovery time

mu = 1.0/(365.0*75.0) # set the birth and death rate

N = 10**3 # Set the population size
 
beta = R0*(gamma+mu) # the contact rate is already determined

ND = 120.0 # set the simulation time in years

I0 = 1 # initial number of infected individuals

X = np.array((N-I0,I0,0)) # initial state

"""
storage arrays
"""
Rate = np.zeros((6)) # propensities

Change = np.zeros((6,3),dtype=int)
Change[0,:]=([-1, 1, 0]) # stoichiometry
Change[1,:]=([0, -1, 1])
Change[2,:]=([1, 0, 0])
Change[3,:]=([-1, 0, 0])
Change[4,:]=([0, -1, 0])
Change[5,:]=([0, 0, -1])


def stoc_eqs(T,X): 
    """
    Gillespie algorithm update
    """
    N = np.sum(X[range(3)])
    Rate[0] = beta*X[0]*X[1]/N
    Rate[1] = gamma*X[1]
    Rate[2] = mu*N
    Rate[3] = mu*X[0]
    Rate[4] = mu*X[1] 
    Rate[5] = mu*X[2]  
    R1=ss.uniform.rvs()
    R2=ss.uniform.rvs()
    ts = -np.log(R2)/(np.sum(Rate))
    j = np.where(np.cumsum(Rate)>=R1*np.sum(Rate))[0][0]
    X[range(3)]+=Change[j,:]
    return [T+ts,X]

def Stoch_Iteration(T0,X0):
    """
    Gillespie algorithm
    """
    lop = 0
    T=[T0]
    S=[X0[0]]
    I=[X0[1]]
    R=[X0[2]]
    while T[lop] < ND:
        lop=lop+1
        [T0,X0] = stoc_eqs(T0,X0)
        T.append(T0)
        S.append(X0[0])
        I.append(X0[1])
        R.append(X0[2])        
    return [T,S,I,R]    

if __name__=="__main__":
    
    
    [T,S,I,R]=Stoch_Iteration(0,X)    
    
    fig,ax = plt.subplots(3,1,sharex=True)
    ax[0].plot(T, S, 'g')
    ax[0].set_ylabel ('Susceptible')
    ax[1].plot(T, I, 'r')
    ax[1].set_ylabel ('Infectado')
    ax[2].plot(T, R, 'k')
    ax[2].set_xlabel ('Tiempo (días)')
    ax[2].set_ylabel ('Recuperado')
