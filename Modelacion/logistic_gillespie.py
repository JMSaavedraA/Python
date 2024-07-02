#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 11:14:10 2022

@author: marcos

"""

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
plt.style.use('seaborn-talk')

"""
Simula el crecimiento de una población usando el algoritmo Gillespie
"""

class logistic():

    def __init__(self, b1=3, b2=1.0/1000.0, d1=1, d2=1.0/1000.0, n0=1, ti=0.0, tf=60.0):
        """
        Inicializa la clase logistic. Define la tasa de crecimiento,
        la capacidad de carga, la condicion inicial y el tiempo de
        integracion
        
            Parameters
            ----------
            b1 : TYPE. float
                DESCRIPTION. Parte constante de la tasa de nacimiento
            b2 : TYPE. Float
                DESCRIPTION. Parte lineal de la tasa de nacimiento
            d1 : TYPE. Float
                DESCRIPTION. Parte constante de la tasa de muete
            d2 : TYPE. Float
                DESCRIPTION. Parte lineal de la tasa de muerte
            n0 : TYPE. Float
                DESCRIPTION. Poblacion inicial
            ti : TYPE. Float
                DESCRIPTION. Tiempo inicial
            tf : TYPE. Float
                DESCRIPTION. tiempo final               
                
            Returns
            -------
            None.
                
        """
        self.b1 = b1
        self.b2 = b2
        self.d1 = d1
        self.d2 = d2        
        self.n0 = n0
        self.ti = ti
        self.tf = tf  
        
        
    def step(self, x, ts):
        """
        Hace un paso del algoritmo Gillespie

        Parameters
        ----------
        x : TYPE. Float
            DESCRIPTION. Poblacion
        ts : TYPE. Float
            DESCRIPTION. tiempo

        Returns
        -------
        [x,t] : TYPE. List of floats
            DESCRIPTION. Poblacion y tiempo actualizados

        """
        Rate1 = (self.b1-self.b2*x)*x
        Rate2 = (self.d1+self.d2*x)*x
        R1=ss.uniform.rvs()
        R2=ss.uniform.rvs()
        ts += -np.log(R2)/(Rate1+Rate2)
        if R1<(Rate1/(Rate1+Rate2)):
            x += 1  # birth
        else:
            x -= 1  # death
        return [x,ts]

    def gillespie(self):
        """
        Algoritmo Gillespìe

        Returns
        -------
        [X,T] : TYPE. List of arrays
            DESCRIPTION. Estados y tiempos de la realizacion
        """
        
        lop = 0
        ts = self.ti
        T = [self.ti]
        X = [self.n0]        
        while T[lop]<self.tf and X[lop]>0:
            [x,ts] = self.step(X[lop],T[lop])
            T.append(ts)
            X.append(x)
            lop += 1
        return [X,T]

if __name__ == '__main__':  
    
    run_logistic = logistic() # inicializa la clase
    [X,T] = run_logistic.gillespie() # haz una realizacion del proceso
    fig,ax = plt.subplots(1,1)
    ax.plot(T, X, 'r') # Haz una gráfica de la realizacion
    ax.set_xlabel ('Tiempo (días)')
    ax.set_ylabel ('Población')    
        
