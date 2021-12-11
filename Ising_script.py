# -------------------------------------------------------------------
# Code developed by A. Caio and G. Rizzi; 
# University of Bologna (IT).
# -------------------------------------------------------------------
# This python module contains functions and a class (IsingSystem_2D) 
# used for Metropolis Monte Carlo simulation of a 2D Ising lattice.
#
# -------------------------------------------------------------------# 

import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import newton
from scipy.optimize import curve_fit
import pandas as pd
from tkinter import*

## Fitting functions
#--------------------------------------------------------------------
def gaussian(x,y0,A,x0,width): 
    return y0 + A*np.exp(-(x-x0)**2/(2*width**2)) 

def power(x,A,x0,alpha):
    return A*(abs(x0 - x))**alpha

def linear(xdata,a,b):
    return a + b*xdata

def fit_data_to_function(
    x, y, function, guess = None, plot=True
):
    params, covariance = curve_fit(function, x, y, guess)

    y_fit = function(x, *params)
    if plot:    
        plt.plot(x, y, ".", label="Points",lw = 4, color = "red")
        plt.legend(fontsize = 13)
        plt.show()
    return params, covariance   

## class
#-------------------------------------------------------------------

class IsingSystem_2D:
    '''This class performs the 2D Ising system simulation by means of 
       the Metropolis Monte Carlo algorithm. 
       ------------------------------------------------------------
       
       Arguments:
       size: lattice dimension (NXN)
       exchange: intensity of the exchange interaction term
       external_field: intensity of the external magnetic field
       m_steps: Metropolis Monte Carlo steps
       init_config: parameter that determines how the initial state has to be computed; 
                    if 0: random spins thoughout the lattice (hot condition)
                    if 1: all the spins point up (cold condition)
                    if -1: all the spins point down (cold condition)
       temperature: heat bath temperature.
       
       '''
     
    def __init__(self, size, exchange, temperature, external_field, m_steps, init_config):
        
        self.N = size
        self.J = exchange
        self.B = external_field
        self.MC_steps = m_steps
        
        self.Kb = 1   # Botzmann constant
        self.u_b = 1  # bohr magneton
        
        self.T = temperature
        self.generate_lattice(init_config)
        
        
  
    def generate_lattice(self, init_config):
    
        """
        Generating a 2D lattice with a random spin configuration, 
        
        INPUT:   init_config == 0: random initial lattice
                 init_config == 1: hot initial lattice
                 init_config == -1: cold initial lattice
                 
        OUTPUT:  lattice: NxN array with each entry in the array being +1 or -1,
        representing the direction of the spin of the particle on that lattice site
        
        """
        if init_config == 0:
            self.lattice = np.random.choice([-1, 1], (self.N, self.N))
            self.equilibration()
            
        if init_config == 1:
            self.lattice = np.empty((self.N,self.N))
            self.lattice[:,:] = 1
            self.equilibration()
        
        if init_config == -1:
            self.lattice = np.empty((self.N,self.N))
            self.lattice[:,:] = -1
            self.equilibration()


    def site_E_int(self, site):
        """
        Finding the contribution to the total energy from a single lattice site
        
        Argument: 
        site: a list that contains the positions in x and y of the lattice site.
        
        """
        #Extracting x and y indices
        x = site[0]
        y = site[1]
        
        #Interaction term  (using % operator to apply periodic boundary conditions)
        interaction_E = - self.J * self.lattice[x, y] *(
            self.lattice[(x + 1)% self.N, y] + 
            self.lattice[x, (y + 1) % self.N] + 
            self.lattice[(x -  1)% self.N, y] + 
            self.lattice[x, (y - 1)% self.N]
            )
        
        return interaction_E
    
    def site_E_ext(self,site):
        #Extracting x and y indices
        x = site[0]
        y = site[1]
        
        external_E = -self.B* self.u_b * self.lattice[x, y] 
        return external_E
    

    def init_en(self):
        """
        Finding the total lattice energy energy associated 
        with the lattice in a specific microstate
        
        """    
        E = 0
        for i in range(self.N):
            for j in range(self.N):
                E += self.site_E_int([i,j])/2 + self.site_E_ext([i,j])
        return E
    
    
    def init_magn(self):
        """
        Finding the overall magnetisation associated
        with the lattice in a specific microstate
        
        """
        M = np.sum(self.lattice)
        return M
        
    
    def flip(self):
        """
        Sweeping though the lattice and, if Metropolis condition is met, flipping the spin.
        
        """
        for ix in range(self.N):
            for iy in range(self.N):
                delta_E = -2 * (self.site_E_int([ix, iy])+self.site_E_ext([ix,iy])) #Energy difference between flipped and unflipped spin    
                if (delta_E < 0) or (np.exp(-(delta_E)/(self.Kb * self.T)) > np.random.rand()):                                              
                    self.lattice[ix,iy] *= -1
                    self.Ene += delta_E
                    self.Magn += 2*self.lattice[ix,iy]

            
    def equilibration(self):
        """
        Performing the initial equilibration at temperature T.
        """
        self.Ene = self.init_en()
        self.Magn = self.init_magn()
        for n_steps in range(self.MC_steps):
            self.flip()
            
    
    def correlation(self):
        """
        Computing the correlation function G_glob
        """
        self.G_glob= np.empty(int(self.N/2)) #correlation global function
        
        G_loc=np.empty(int(self.N/2))
        
        for r in range(1,int(self.N/2)+1):
            for n_steps in range(1, self.MC_steps + 1):
                
                self.flip()
                
                temp=0
                for i in range(self.N):
                    for k in range(self.N):
                        temp2 = self.lattice[i, k] *(
                                self.lattice[(i + r)% self.N, k] + 
                                self.lattice[i, (k + r) % self.N] + 
                                self.lattice[(i -  r)% self.N, k] + 
                                self.lattice[i, (k - r)% self.N]
                                   )      
                        temp += temp2/4

                G_loc[r-1] += temp/self.N**2
              
            self.G_glob[r-1] = G_loc[r-1]/self.MC_steps

    
    def metropolis_MC(self):
        
        """
        Implementing the Metropolis algorithm (Monte Carlo), finding energy and
        magnetisation of the system after each sweep and finding the average
        energy and magnetisation over all sweeps
        
        """
        
        #Array containing lattice configuration after each Metropolis sweep
        #(dimensions: (num_steps + 1) x N x N)
        self.lattice_list = np.zeros([self.MC_steps + 1, self.N, self.N])   
        self.lattice_list[0] = self.lattice  
        
        #Array containing energy associated with lattice configuration after each
        #Metropolis sweep
        #(dimensions: (num_steps + 1) x 1)
        self.energy_list = np.zeros(self.MC_steps + 1)
        self.energy_list[0] = self.Ene        
        
        #Array containing magnetisation associated with lattice configuration after
        #each Metropolis sweep
        #(dimensions: (num_steps + 1) x 1)
        self.magnetisation_list = np.zeros(self.MC_steps + 1)
        self.magnetisation_list [0] = self.Magn
        
        for j in range(1, self.MC_steps + 1):
            
            #Looping through lattice sites in order
            self.flip()
            
            #Assigning updated values
            self.lattice_list[j] = self.lattice
            self.energy_list[j] = self.Ene
            self.magnetisation_list[j] = self.Magn
            
        
        self.av_E = np.sum(self.energy_list)
        self.av_M = np.sum(self.magnetisation_list)
        

        self.av_E_2 = np.sum(np.square(self.energy_list))
        self.av_M_2 = np.sum(np.square(self.magnetisation_list))
   
       
        
    def physics_val(self, T_array):
        
        self.E_xspin = np.zeros(len(T_array))
        self.M_xspin = np.zeros(len(T_array))
        self.C = np.zeros(len(T_array))
        self.Chi = np.zeros(len(T_array))
        
        M_2 = np.zeros(len(T_array))
        E_2 = np.zeros(len(T_array))
        
                    
        n1 = 1/(self.N**2 * (self.MC_steps + 1))
        n2 = 1/(self.N**2 * (self.MC_steps + 1)**2)
        
        
        for k,T in enumerate(T_array):
            
            self.T = T
            self.metropolis_MC()
            
            self.E_xspin[k] = self.av_E*n1
            self.M_xspin[k] = self.av_M*n1
            self.C[k] = (self.av_E_2*n1 - n2*self.av_E**2)/(self.Kb*T**2)
            self.Chi[k] = (self.av_M_2*n1 - n2*self.av_M**2)/(self.Kb*T)
            
    
    
    def G_corr(self,T_list):
        
        G_matrix = np.empty(int(self.N/2),len(T_list))
        
        x_data = np.linspace(1,int(self.N/2),int(self.N/2) )

        for i,T in enumerate(T_list):
            
            self.T = T
            self.correlation()
            G_matrix[:,i] = self.G_glob[:]
            
        
        return G_matrix, x_data
    
    
    def MvsB(self,T_array, B_list):         
        M_matrix = np.empty((len(T_array),len(B_list)))
    
        for k,B in enumerate(B_list):  
            for i,T in enumerate(T_array):
                self.T = T
                self.B = B
                
                self.metropolis_MC()
                
                M_matrix[i,k] = self.av_M/(self.N**2*(self.MC_steps+1))
        
                                    
        return M_matrix
                     
