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

ND = 100
muH = 1.0/(75.0*365.0)
betaHV = 0.25
gammaH = 1.0/7.0
muV = 1.0/21.0
betaVH = 0.5


SH0 = 100
IH0 = 0
RH0 = 0
SV0 = 199
IV0 = 1
sample_size = 50

Change=np.zeros((10,5),dtype=int) # stoichiometry
Change[0,:]=([1, 0, 0, 0, 0]);#Nacimiento de humano
Change[1,:]=([-1, 1, 0, 0, 0]);#Infección de humano
Change[2,:]=([0, -1, 1, 0, 0]);#Remoción de humano
Change[3,:]=([-1, 0, 0, 0, 0]);#Muerte de humano susceptible
Change[4,:]=([0, -1, 0, 0, 0]);#Muerte de humano infecciso
Change[5,:]=([0, 0, -1, 0, 0]);#Muerte de humano removido
Change[6,:]=([0, 0, 0, 1, 0]);#Nacimiento de mosquito
Change[7,:]=([0, 0, 0, -1, 1]);#Infección de mosquito
Change[8,:]=([0, 0, 0, -1, 0]);#Muerte de mosquito susceptible
Change[9,:]=([0, 0, 0, 0, -1]);#Muerte de mosquito infeccioso

def stoc_eqs(T,X):
    Rate=np.zeros((10))
    Rate[0] = muH*(X[0]+X[1]+X[2]) #Nacimiento de humano
    Rate[1] = betaHV*X[4]*X[0]/(X[0]+X[1]+X[2]) #Infección de humano
    Rate[2] = gammaH*X[1] #Remoción de humano
    Rate[3] = muH*X[0] #Muerte de humano susceptible                   
    Rate[4] = muH*X[1] #Muerte de humano infecciso
    Rate[5] = muH*X[2] #Muerte de humano removido
    Rate[6] = muV*(X[3]+X[4]) #Nacimiento de mosquito
    Rate[7] = betaVH*X[3]*X[1]/(X[0]+X[1]+X[2]) #Infección de mosquito
    Rate[8] = muV*X[3] #Muerte de mosquito susceptible
    Rate[9] = muV*X[4] #Muerte de mosquito infeccioso
    R1=ss.uniform.rvs()
    R2=ss.uniform.rvs()
    ts = -np.log(R2)/(np.sum(Rate))
    j = np.where(np.cumsum(Rate)>=R1*np.sum(Rate))[0][0]
    X[range(5)]+=Change[j,:]    
    return [T+ts,X]


def Stoch_Iteration(T0,X0):
    lop = 0
    T = [T0]
    SH = [X0[0]]
    IH = [X0[1]]
    RH = [X0[2]]
    SV = [X0[3]]
    IV = [X0[4]]
    
    while T[lop] < ND:
    #while T[lop] < ND and SH[lop] >= 0 and SV[lop] >= 0:        
        [T0,X0] = stoc_eqs(T0,X0)
        T.append(T0)
        SH.append(X0[0])
        IH.append(X0[1])
        RH.append(X0[2])
        SV.append(X0[3])
        IV.append(X0[4])
        lop += 1        
    return [T,SH,IH,RH,SV,IV]



fig,ax = plt.subplots(1,1)

for k in np.arange(sample_size):
    X0 = np.array([SH0,IH0,RH0,SV0,IV0])
    [T,SH,IH,RH,SV,IV]=Stoch_Iteration(0,X0)    
    ax.plot(T, IH, 'r', lw=0.5, alpha=0.5)
ax.set_xlabel ('Tiempo (Días)')
ax.set_ylabel ('Humanos Infectados')


