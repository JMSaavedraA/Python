import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-talk')
import pytwalk
from scipy import integrate
import corner

""" Fix the seed of fandom number generation """
np.random.seed(2022)


class direct_problem:
    """
    Simulate the SIRS model of: 
    Weber, Andreas, Martin Weber, and Paul Milligan. 
    "Modeling epidemics caused by respiratory syncytial virus (RSV)." 
    Mathematical biosciences 172, no. 2 (2001): 95-113.
    """
    
    def __init__(self,nu=36,gamma=1.8,which_country='gambia'):
        """
        Set non-transmission SIRS model parameters
        
        Parameters
        ----------
        mu : TYPE, float
            DESCRIPTION. Birth and death rate.
        nu : TYPE, optional
            DESCRIPTION. Loss of immunity rate.
        gamma : TYPE, float
            DESCRIPTION. Recovery rate.
        N : TYPE, float
            DESCRIPTION. Population size.

        Returns
        -------
        None.

        """
        self.nu=nu
        self.gamma=gamma
        self.which_country = which_country
        if self.which_country=='gambia':
            self.mu = 0.041
            self.N=736
            self.data = np.loadtxt('gambia.txt')
        elif self.which_country=='finland':
            self.mu = 0.013
            self.N=2420
            self.data = np.loadtxt('finland.txt')[9:]                    
        self.n = len(self.data)
        self.tdata = np.linspace(0,(self.n - 1.0)/12.0,self.n)      

    def rhs(self,x,t,p):
        """
        Right hand side of the SIRS epidemic model.
        State variables are divided by the system size. 
        System is scaled in years.

        Parameters
        ----------
        t : TYPE, float
            DESCRIPTION. Time
        x : TYPE, array of floats
            DESCRIPTION. State
        p : TYPE, array of floats
            DESCRIPTION. parameters

        Returns
        -------
        fx : TYPE
            DESCRIPTION. Dynamical system right hand side

        """
        betat = p[0]*(1.0+p[1]*np.cos(2.0*np.pi*t+p[2]))*x[0]*x[1]
        f0 = self.mu-self.mu*x[0]-betat+self.gamma*x[2]
        f1 = betat-self.nu*x[1]-self.mu*x[1]
        f2 = self.nu*x[1]-self.mu*x[2]-self.gamma*x[2]
        f3 = betat
        return np.array([f0,f1,f2,f3])
    
    def solve(self,p):
        """
        Propose an initial condition. Evolve the dynamical system 100 years.
        Use the endpoint as initial condition to simulate the system in the interval
        where there is data.

        Parameters
        ----------
        p : TYPE, array of floats
            DESCRIPTION. Vector of parameters

        Returns
        -------
        ySoln : TYPE, Array of floats
            DESCRIPTION. Solution of the SIRS model.

        """

        y0 = np.array([1.0-p[3]-p[4],p[3],p[4],sirs.data[0]])
        y0 = integrate.odeint(self.rhs,y0,np.linspace(0.0,100.0,1000),args=(p,))[-1,:]
        ySoln = integrate.odeint(self.rhs,y0,self.tdata,args=(p,))
        return ySoln        
    
if __name__=="__main__":
    sirs = direct_problem(which_country='gambia')
    
    def support(p):
        """
                

        Parameters
        ----------
        p : TYPE, array of floats
            DESCRIPTION. Vector of parameters

        Returns
        -------
        rt : TYPE, Booloean
            DESCRIPTION. True if point lies in the support
        """
        rt = True
        rt &= (30.0<p[0]<70.0)
        rt &= (0.1<p[1]<0.5)
        rt &= (0.0<p[2]<2.0*np.pi)        
        rt &= (0.0<p[3]<1.0)
        rt &= (0.0<p[4]<1.0)
        rt &= (0.0<p[5]<1.0)        
        return rt        
    
    def init():
        """
        Draw a random point in the support

        Returns
        -------
        p : TYPE
            DESCRIPTION.

        """
        p = np.zeros(6)
        if sirs.which_country=='gambia':
            p[0] = 60.0+ss.norm.rvs(loc=0,scale=0.01)
            p[1] = 0.16+ss.norm.rvs(loc=0,scale=0.01)
        elif sirs.which_country=='finland':
            p[0] = 44.0+ss.norm.rvs(loc=0,scale=0.01)
            p[1] = 0.36+ss.norm.rvs(loc=0,scale=0.01)        
        p[2] = np.random.uniform(low=0.0,high=2.0*np.pi)       
        p[3] = np.random.uniform(low=0.0,high=1.0)       
        p[4] = np.random.uniform(low=0.0,high=1.0)
        p[5] = np.random.uniform(low=0.0,high=1.0)        
        return p
        
    def energy(p):
        """
        Minus the logarithm of the posterior distribution.        

        Parameters
        ----------
        p : TYPE, vector of floats
            DESCRIPTION. Vector of parameters

        Returns
        -------
        -log(pi(p|y)*pi(p) TYPE, float
            DESCRIPTION. Minus the logarithm of the posterior distribution.
        """

        log_prior = 0.0
        log_prior += ss.gamma.logpdf(p[0],1.0,scale=np.sqrt(1.5*(sirs.nu+sirs.mu)))
        log_prior += ss.beta.logpdf(p[1],1.1,1.1)
        log_prior += ss.beta.logpdf(p[2],1.1,1.1,scale=2.0*np.pi)
        log_prior += ss.beta.logpdf(p[3],1.1,1.1)
        log_prior += ss.beta.logpdf(p[4],1.1,1.1)
        log_prior += ss.beta.logpdf(p[5],1.1,1.1)        
        log_likelihood = 0.0
        mu = p[5]*np.diff(sirs.N*sirs.solve(p)[:,-1])
        omega = 1.0
        theta = 2.0
        r = mu/(omega-1.0+theta*mu)
        q = 1.0/(omega+theta*mu)
        log_likelihood = np.sum(ss.nbinom.logpmf(sirs.data[1:], r, q)) # negative binomial        
        #log_likelihood += np.sum(ss.poisson.logpmf(sirs.data[1:],mu)) # poisson
        #log_likelihood += np.sum(ss.norm.logpdf(sirs.data[1:],loc=mu,scale=20.0)) # normal        
        print(-log_likelihood - log_prior)
        return -log_likelihood - log_prior
    
    """ run the twalk """
    syncytial = pytwalk.pytwalk(n=6,U=energy,Supp=support)
    syncytial.Run(T=50000,x0=init(),xp0=init())
            
    """ trace plot """
    burnin=25000
    subsample=100
    plt.figure()
    syncytial.Ana(start=burnin)
    
    """ corner """
    samples = syncytial.Output[burnin::subsample,:-1]
    quantiles = np.quantile(samples,q=[0.01,0.99],axis=0)
    plot_range = [(quantiles[0,x],quantiles[1,x]) for x in np.arange(6)]
    labels =  [r"$b_0$", r"$b_1$",r"$\phi$",r"$I(0)$", r"$R(0)$",r"$q$"]
    corner.corner(samples,
                  labels=labels,
                  range = plot_range,
                  plot_datapoints=False,
                  show_titles=True
                  )

    """ prediction vs data """
    solns = np.zeros((100,len(sirs.data)))
    for index in np.arange(100):
        p = syncytial.Output[index*subsample,:-1]
        mu = p[5]*np.diff(sirs.N*sirs.solve(p)[:,-1])
        solns[index,0] = sirs.data[0]
        solns[index,1:] = mu
    
    solns_median = np.median(solns,axis=0)
    solns_std = np.std(solns,axis=0)    
    solns_quant = np.quantile(solns,q=[0.05,0.95],axis=0)     

    fig,ax = plt.subplots(1,1)
    ax.plot(sirs.tdata,sirs.data,'ro-',label='data')
    ax.plot(sirs.tdata,solns_median,'k',label='median')
    ax.fill_between(sirs.tdata,solns_quant[1,:],solns_quant[0,:],color='k',alpha=0.25,label='0.05-0.95 probability')
    plt.legend()