# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 18:20:45 2020

@author: Antonio Frederico Nesti Lopes
"""

import numpy as np
from numpy.random import normal
from numpy.random import uniform
from scipy.stats import norm
#from mpl_toolkits.mplot3d import Axes3D
#import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection#,PolyCollection
#from matplotlib.colors import colorConverter

# This is the function that defines the distribution we want to sample from
# GMM
def p(x,a1=0.5,a2=0.5,m1=0,s1=1,m2=3,s2=0.5):
    '''
    This function is for the Metropolis Hastings algorithm
    '''    

    r = a1*(s1*norm.pdf(x)+m1) + a2*(s2*norm.pdf(x)+m2)
    return r

def Sample_Conditional_Normal(mean,cov,conditional_data):
    '''
    This function is for the Metropolis Hastings algorithm
    '''    

    c11 = cov[0, 0] 
    c12 = cov[0, 1] 
    c21 = cov[1, 0] 
    c22 = cov[1, 1] 
    
    m1 = mean[0].T
    m2 = mean[1].T 
    
    #conditional_data = normal(m2, c22, 1)
    
    # Doing Schur complement
    conditional_mu = m1 + c12*(1/c22)*(conditional_data - m2) # check this
    conditional_cov = c11 + c12*(1/c22)*c21 # check this
    
    dependent_data = normal(conditional_mu, conditional_cov, 1)
    
    dens_data = conditional_cov*norm.pdf(dependent_data) + conditional_mu
    
    return [dependent_data,dens_data]

def A(mean,cov,x_s,x_i):
    '''
    This function is for the Metropolis Hastings algorithm
    '''    
    q_is = Sample_Conditional_Normal(mean,cov,x_s)[0]
    q_si = Sample_Conditional_Normal(mean,cov,x_i)[0]
    
    p_s = p(x_s)
    p_i = p(x_i)
    
    mc_kernel_num = p_s*q_is
    mc_kernel_denom = p_i*q_si
    mc_kernel = mc_kernel_num/mc_kernel_denom
    
    return min(1,mc_kernel)

class Metropolis_Hastings():
    
    def __init__(self,N,sigma_prop):
        self.samples = [1000]
        self.num_samples = N
        self.var_prop = sigma_prop
        
    def MCMC(self):
        
        for i in range(self.num_samples):
            
            # parameters for conditional normal sampling
            mean = np.array([0, 0])
            cov = np.array(
                [[ self.var_prop,  0.0], 
                 [ 0.0,  self.var_prop]])  
            
            u = uniform(0,1,1)
            x_star = Sample_Conditional_Normal(mean,cov,self.samples[-1])[0]
            
            if u < A(mean,cov,self.samples[-1],x_star):
                self.samples.append(x_star)
            else:
                self.samples.append(self.samples[-1])
            
    def plot(self):
        #print(self.samples)
        plt.plot(self.samples[1:])
        
###############################################################################

def model(t):
    '''
    This function is for the Bootstrap Filter
    '''
    #Noises
    v_t = normal(0,10,1)
    w_t = normal(0,1,1)
    
    if t == 0:
        x_t = normal(0,10,1)
        y_t = (x_t**2)/2 + w_t
                
    else:
        
        x_prev = model(t-1)[0]
        
        x_t = 0.5*x_prev + 25*(x_prev/(1+(x_prev**2))) + 8*np.cos(1.2*t) + v_t
        y_t = (x_t**2)/2 + w_t
        
    return [x_t,y_t]

# Talvez não precise de duas funções model

def model_RStep(x_prev,t):
    '''
    x_prev here is different than the previous function
    '''
    #Noises
    v_t = normal(0,10,1)
    w_t = normal(0,1,1)
    
    x_t = 0.5*x_prev + 25*(x_prev/(1+(x_prev**2))) + 8*np.cos(1.2*t) + v_t
    y_t = (x_t**2)/2 + w_t
        
    return [x_t,y_t]
    
        
class Bootstrap_Filter():
    '''
    The notation x_0t means {x0,x1,...,xt} = x_0:t
    While w_t = {wt}
    '''
    
    def __init__(self,N,total_times):
        
        self.num_samples = N
        self.T = total_times
        self.t = 0
        self.xtil_0t = np.zeros((self.T,self.num_samples))
        self.x_resample = np.zeros((self.T,self.num_samples))
        self.wtil_t = np.zeros((1,self.num_samples))
        
    def SMC(self):

        # Init
        for i in range(self.num_samples):
            self.xtil_0t[self.t,:] = model(self.t)[0].tolist()[0]
        self.t += 1
        
        # Repeat
        while self.t < self.T:
            
            # IS Step + Prediction (variety intro):
            for i in range(self.num_samples):
                self.xtil_0t[self.t,:] = model(self.t)[0].tolist()[0]
                self.wtil_t[0,:] = model(self.t)[1].tolist()[0]
                
                self.wtil_t[0,:] /= np.sum(self.wtil_t[0,:]) 
                
            # Selection Step (resample):
            for timestep in range(self.t):
                
                obj_resample = []
                for n in range(self.num_samples):
                    obj_resample.append(model_RStep(self.xtil_0t[timestep,n],timestep)[0].tolist()[0])
                    
                resample = np.random.choice(np.array(obj_resample),size=self.num_samples,replace=True,p=self.wtil_t[0,:])
                self.x_resample[timestep,:] = resample
                
                
            # Re-assign  x:
            self.xtil_0t[:] = self.x_resample[:]
            
            print('Time step: ',self.t)
            self.t += 1    
            
    
    def plot(self):
        
        fig = plt.figure()

        ax = fig.gca(projection='3d')
        
        zs = [t for t in range(self.T)]
        
        verts=[list(zip(np.histogram(self.xtil_0t[t,:],density=True,bins=10)[1][:-1], np.histogram(self.xtil_0t[t,:],density=True,bins=10)[0])) for t in range(self.T)]
        #verts=[list(zip(np.linspace(-25,25,self.num_samples), np.histogram(self.xtil_0t[t,:],density=True,bins=10)[0])) for t in range(self.T)]
                
        poly = LineCollection(verts)
        #poly = LineCollection([np.histogram(self.xtil_0t[t,:],density=True,bins=10) for t in range(self.T)-1])
        #poly.set_alpha(0.7)
        ax.add_collection3d(poly, zs=zs, zdir='y')
        
        ax.set_xlim3d(-50, 50)
        ax.set_ylim3d(0, self.T)
        ax.set_zlim3d(0, 0.04)

        plt.show()
    
    

    
    
    
    
