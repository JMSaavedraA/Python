#!/usr/bin/env python3
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
plt.style.use('seaborn-talk')

c1 = 1.0 
c2 = 0.1
c3 = 1.5
c4 = 0.75

ND=50.0
P0=20
D0=10

X0 = np.array([P0,D0])

Rate = np.zeros((4))
Change = np.zeros((4,2),dtype=int)

Change[0,:]=[1, 0]
Change[1,:]=[-1, 0]
Change[2,:]=[0, 1]
Change[3,:]=[0, -1]  

def stoc_eqs(T,X): 
    Rate[0] = c1*X[0]
    Rate[1] = c2*X[0]*X[1] 
    Rate[2] = c2*c4*X[0]*X[1] 
    Rate[3] = c3*X[1]  
    R1=ss.uniform.rvs()
    R2=ss.uniform.rvs()
    ts = -np.log(R2)/(np.sum(Rate))
    j = np.where(np.cumsum(Rate)>=R1*np.sum(Rate))[0][0]
    X[range(2)]+=Change[j,:] 
    return [T+ts,X]

def Stoch_Iteration(T0,X0):
    lop=0
    T=[T0]
    P=[X0[0]]
    D=[X0[1]]
    while T[lop] < ND and P[lop]>0 and D[lop]>0:
        lop=lop+1                
        [T0,X0] = stoc_eqs(T0,X0)
        T.append(T0)
        P.append(X0[0])
        D.append(X0[1])
    return [T,P,D]

[T,P,D]=Stoch_Iteration(0,X0)

rabbits, foxes = P,D
f1 = plt.figure()
plt.plot(T, rabbits, 'r-', label='Rabbits')
plt.plot(T, foxes  , 'b-', label='Foxes')
plt.grid()
plt.legend(loc='best')
plt.xlabel('time')
plt.ylabel('population')
plt.title('Evolution of fox and rabbit populations')
plt.axis('tight')
f1.savefig('gillespie_rabbits_and_foxes_1.png')

rabbits, foxes = P,D
f2 = plt.figure()
plt.plot(rabbits,foxes)
plt.plot(rabbits[0],foxes[0],'og')
plt.plot(rabbits[-1],foxes[-1],'or')
plt.grid()
plt.xlabel('Number of foxes')
plt.ylabel('Number of rabbits')
plt.title('Trajectories')
f2.savefig('gillespie_rabbits_and_foxes_2.png')