#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 13:09:06 2022

@author: marcos
"""
import numpy as np
import scipy.stats as ss
from scipy import integrate
import matplotlib.pyplot as plt
plt.style.use('seaborn-talk')

class logistic():
    
    def __init__(self, b1=3, b2=1/100, d1=1, d2=1/100, n0=1, ti=0, tf=60):
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
        self.r = b1-d1 # calcula la tasa de crecimiento para b1 diferente de d1
        self.k = self.r/(b2+d2) # calcula la capacidad de carga
        self.n0 = (n0,) #EL ESTADO INICIAL ES UNA TUPLA
        self.teval = np.linspace(0.0,20.0,21)
        
    def f(self, t, x):
        """
        Lado rerecho de la ecuacion logistica
        
        Parameters
        ----------
        t : TYPE. Float
            DESCRIPTION. Tiempo
        x : TYPE. Float
            DESCRIPTION. Poblacion

        Returns
        -------
        fx : TYPE. Float
            DESCRIPTION. Lado derecho de la ecuacion diferencial

        """
        fx = self.r*x*(1-x/self.k)
        return fx
        
    def solve(self):
        ti = self.teval[0]
        tf = self.teval[-1]+0.5
        return integrate.solve_ivp(self.f, (ti,tf), self.n0, method='RK45', 
                                   t_eval = self.teval)
        

if __name__ == '__main__':

    run_logistic = logistic()   # inicializa la clase     
    soln = run_logistic.solve() # resuelve el problema de valores iniciales
    time = run_logistic.teval
    data = soln.y.flatten() #+ ss.norm.rvs(scale=0.1,size=len(run_logistic.teval)) # suma ruido a los datos
    fig,ax = plt.subplots(1,1)
    ax.plot(time,data,'ok')
    ax.set_xlabel ('Tiempo (días)')
    ax.set_ylabel ('Población')      