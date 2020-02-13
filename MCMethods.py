# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 18:20:45 2020

@author: Antonio Frederico Nesti Lopes
"""

import numpy as np
from numpy.random import normal
from numpy.random import uniform
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
from matplotlib.collections import LineCollection#,PolyCollection
from matplotlib.colors import colorConverter
import math
import scipy.stats as ss

# This is the function that defines the distribution we want to sample from
# GMM
def p_real(x,a1=0.5,a2=0.5,m1=0,s1=1,m2=3,s2=0.5):
    '''
    This function is for the Metropolis Hastings algorithm
    '''    
    return a1*norm.pdf(x,loc=m1-1,scale=s1) + a2*norm.pdf(x,loc=m2-1,scale=s2) 

def p(x,a1=0.5,a2=0.5,m1=0,s1=1,m2=3,s2=0.5):
    '''
    This function is for the Metropolis Hastings algorithm
    unnormalized target distribution
    '''    
    g1 = np.exp(-((x-m1)**2)/(2*(s1**2)))
    g2 = np.exp(-((x-m2)**2)/(2*(s2**2)))
    return a1*g1 + a2*g2
    #return a1*norm.pdf(s1*x+m1) + a2*norm.pdf(s2*x+m2)

class Metropolis_Hastings():
    
    def __init__(self,N,sigma_prop):
        self.samples = [-5]
        self.num_samples = N
        self.var_prop = sigma_prop
  
    def A(self,x_s):
        
        q_is = norm.pdf(self.samples[-1], loc=x_s-1, scale=np.sqrt(self.var_prop))
        q_si = norm.pdf(x_s, loc=self.samples[-1]-1, scale=np.sqrt(self.var_prop))
        
        mc_kernel = (p(x_s)*q_is)/(p(self.samples[-1])*q_si)
        
        return min(1,mc_kernel)
        
    def MCMC(self):
        for i in range(self.num_samples):
            u = uniform(0,1)
            x_star = random.gauss(self.samples[-1],np.sqrt(self.var_prop))
            
            if u < self.A(x_star):
                self.samples.append(x_star)
            else:
                self.samples.append(self.samples[-1])
            
    def plot(self):
        plt.figure()
        plt.plot(np.array(self.samples[1:]))
        plt.suptitle('Random Walk')
        plt.title('For the last 2000 samples')
        plt.xlim(len(self.samples[1:])-2000,len(self.samples[1:]))
        plt.show()
        #plt.figure()
        
        fig, ax = plt.subplots()
        
        sns.distplot(np.array(self.samples[1:]), hist=True, kde=True, 
             bins=int(90/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2},ax=ax)
        #plt.show()
        #plt.figure()
        g = sns.lineplot(np.arange(-4,6,0.001),p_real(np.arange(-5,5,0.001)),ax=ax,color='r')
        plt.setp(g.lines[1],linewidth=4)
        plt.title('Metropolis-Hastings Monte Carlo')
        #params = (self.num_samples,self.var_prop)
        #plt.title('(No. Simul,Proposal Var) =(',params)
        plt.show()
        del(self.samples)
        
###############################################################################

def xt(x_prev,t):
    return [0.5*x + 25*(x/(1+(x**2))) + 8*np.cos(1.2*t) for x in x_prev]
def yt(xt):
    return [(x**2)/2 for x in xt]

        
class Bootstrap_Filter():
    
    def __init__(self,N,total_times):
        self.num_samples = N
        self.T = total_times
        self.t = 0
        self.xtil_0t = np.zeros((self.T,self.num_samples))
        #self.x = np.zeros((self.T,self.num_samples))
        self.wtil_t = np.zeros((self.num_samples))
        #self.m = []
        
    def SMC(self):

        # Init
        self.xtil_0t[0,:] = norm.rvs(0,math.sqrt(10),self.num_samples)
        self.t += 1
        
        # Repeat
        while self.t < self.T:
            print('Time step: ',self.t)
            
            # IS Step + Prediction (variety intro):
            for i in range(self.num_samples):
                self.xtil_0t[self.t,i] = norm.rvs(xt([self.xtil_0t[self.t-1,i]],self.t),math.sqrt(10)) 
            for i in range(self.num_samples):
                self.wtil_t[i] = norm.pdf(yt([self.xtil_0t[self.t,i]])+norm.rvs(0,1),loc=yt([self.xtil_0t[self.t,i]]),scale=1)  
                #self.wtil_t[i] = norm.pdf(yt([self.xtil_0t[self.t-1,i]])+norm.rvs(0,1),loc=yt([self.xtil_0t[self.t,i]]),scale=1)  #
       
            self.wtil_t[:] /= np.sum(self.wtil_t) 
            
            #self.x[self.t,:] = self.xtil_0t[self.t,:]
            
            #self.m.append(np.sum(self.wtil_t*self.xtil_0t[self.t,:]))
            
            # Selection Step (resample):
            self.xtil_0t[self.t,:] = np.random.choice(self.xtil_0t[self.t,:],size=self.num_samples,replace=True,p=self.wtil_t)           
            
            #self.m.append(np.mean(self.xtil_0t[self.t,:]))
            
            self.t += 1    
            
    def plot(self):   
        for t in range(0,self.T): #1
            plt.figure()
            sns.distplot(self.xtil_0t[t,:], hist=False, kde=True, color = 'darkblue', hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 1})
            plt.ylim(0,0.4)
            plt.xlim(-25,25)
            plt.title('Time:'+str(t))
            plt.show()
        
#        plt.figure()
#        plt.plot(self.m,color='r')
#        plt.plot(np.arange(1,self.T),self.xtil_0t[1:,0],'-o',color='b')
#        plt.suptitle('Prediction')
#        plt.title('Without Smoothing')
#        plt.show()
#        
#        plt.figure()
#        plt.plot((self.m-self.xtil_0t[1:,0])/self.xtil_0t[1:,0])
#        plt.title('Error Perc.')
#        plt.ylim(-2,2)
#        plt.show()
#
#        plt.figure()
#        plt.plot(self.m-self.xtil_0t[1:,0])
#        plt.title('Error')
#        plt.show()
    

    
    
    
    
