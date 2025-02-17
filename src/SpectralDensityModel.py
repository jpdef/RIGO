#Desc : Library for Internal Wave Spectrum Modeling
#Auth : J. DeFilippis
#Date : 11-12-2024
import sys
import numpy as np 
import scipy
import functools

import ModeSolver
from tqdm import tqdm
from functools import partial
tocph = 3600/(2*np.pi)

"""
Garrett Munk Meta Parameters 
"""
class GM81Params:
    """
    Meta parameters for Garrett-Munk model 
        E  energy scale constant
        b  depth of thermocline
        No maximum (surface) stratification
    """
    def __init__(self):
        self.Ec = 6.3e-5    # Dimensionaless empircal scale of IW energy in the ocean
        self.b = 1.3e3     # Vertical scale of stratification
        self.No= 3/(tocph) # Initial Stratification
        self.jstar = 3     # Mode spectrum roll off
        self.N = lambda z : self.No*np.exp(-z/self.b)
        
class CustomParams:
    def __init__(self,Ec,N2,Z):
        self.Ec = Ec
        self.No, self.b = stratification_fit(N2,Z)
    
    def stratification_fit(self,N2,Z):
        def func(x, a, b, c):
            return a * np.exp(-b*x) + c
        N = np.sqrt(N2)
        popt,pcov = curve_fit(func,Z,N,p0=[np.max(N),Z[len(Z)//2],0])
        return popt[0],popt[1]

    
      
class GarrettMunkSpectralModel:
    """
    Internal wave power spectrum using Garrett Munk Parameterization.

        @param N2         : stratification  in [rad/s] nx1 
        @param Z          : depth in [m] nx1
        @param latitude   : local line of latitude in [deg]
        @param jstar      : parameters for IW mode power distribution
        @param max_modes  : number of modes to include in spectrum
    """
    def __init__(self,smparams,latitude=30,max_modes=10,N=None): 
        self.Ec  = smparams.Ec
        self.b   = smparams.b
        self.No  = smparams.No
        self.N   = smparams.N if N is None else N
        self.max_modes = max_modes
        self.jstar = smparams.jstar
        self.f  = ModeSolver.coriolis(latitude) 
    
    #Modal Spectrum [unit less]
    def H(self,j,max_modes=0):
        if j == 0 : 
            return 1
        elif j < 0:
            print('Mode number must be greater than or equal to 0')
            return 0 
        else:
            mm = self.max_modes if max_modes == 0 else max_modes
            return self.M(mm)*self.h(j)
    
    #Mode normalization
    def M(self,max_mode):
        return 1/np.sum(np.array([self.h(j) for j in range(1,max_mode+1)]))
        
    #Modal Distribution        
    def h(self,j):
        return 1/(j**2 + self.jstar**2)
    
    #Frequency distribution [1/omega]
    def B(self,omega):
        return 2*self.f/(np.pi*omega*np.sqrt(omega**2-self.f**2)) 
    
    #Energy Density [m^2/s]
    def E(self,omega,j,max_modes=0):
        return self.B(omega)*self.H(j,max_modes)*self.Ec
    
    """
    Vertically Integrated PSD Functions
     - Removes the No N or No N^-1 depth scaling by integrating
     - KE = int F_u dz
     - PE = int N^2 F_zeta dz
     - TE = int E dz
     - Useful for normalizing modes 
     - rho_sw * VITE = Energy / unit area
    """
    #Total Power Spectral Density Function [m/s]^2 / [omega]
    def VI_PSD_TE(self,omega,mode=0,max_modes=0):
        return self.b**3 * self.No**2 * self.E(omega,mode,max_modes)

    #Vertically Integrated Potential Energy Density Function
    def VI_PSD_PE(self,omega,mode=0,max_modes=0): 
        return ((omega**2 - self.f**2)/ omega**2) * self.VI_PSD_TE(omega,mode,max_modes)

    #Vertically Integrated Kinetic Energy Density Function     
    def VI_PSD_KE(self,omega,mode=0,max_modes=0):
        return ((omega**2 + self.f**2)/ omega**2) * self.VI_PSD_TE(omega,mode,max_modes)

    """
    PSD Functions
    - Power spectral density functions for horizontal velocity and displacement
    """
    def PSD_ZETA(self,z,omega,mode=0):
        depth_scaling = (1/(self.b*self.No))*(1/self.N(z))
        return depth_scaling * self.VI_PSD_PE(omega,mode)
    
    def PSD_VELO(self,z,omega,mode=0):
        depth_scaling = 1/(self.b*self.No)*(self.N(z))
        return depth_scaling * self.VI_PSD_KE(omega,mode)
    
    """
    Discrete Spectrum Functions
    - Get enery in a specific band of frequencies
    - Generate wave amplitudes for a discrete approximation of the spectrum
    """
    
    def sample_variance_1d(self,S,x,dx):
        #Integrate over the spectrum along one dimension
        x_lo = x - dx/2 ; x_hi = x + dx/2
        band_energy = scipy.integrate.quad(S,x_lo,x_hi)[0]
        return band_energy
    
    def generate_discrete_spectrum(self,omegas,modes,psd_type='PE'):
        """
        Get discete amplitudes for each frequency, mode pair
        that approximates the power in the spectrum
        @ omegas    discrete frequencies
        @ modes     discrete modes
        @ psd_type  type of power spectrum to use 'PE' or 'KE'
        """
        if psd_type =='PE':
            S = self.VI_PSD_PE
        elif psd_type == 'KE':
            S = self.VI_PSD_KE
        else:
            S = self.VI_PSD_TE
        
        domegas = np.diff(omegas)
        domega = domegas[0]
        dho = domega/2
        
        #Grid Needs to be uniform
        assert( np.allclose(domegas,domegas[0]) )
        
        #Lowest frequency band needs to be above f
        assert(np.min(omegas)- domega/2 > self.f)
        
        #Highest frequency band needs to be below No
        assert(np.max(omegas)+ domega/2 < self.No)
        ds = np.zeros(shape=(len(omegas),len(modes)), 
                      dtype=[('o',float),('m',float),('E',float)])
        
        for i,o in enumerate(omegas):
            for k,m in enumerate(modes):
                omega_lo = o - dho ; omega_hi = o + dho
            
                #Integrate over the frequency spectrum curve at a fixed mode
                Sm = functools.partial(S,mode=m)
                band_energy = scipy.integrate.quad(Sm,omega_lo,omega_hi)[0]
                ds[i,k]['E'] = band_energy
                ds[i,k]['o'] = o; ds[i,k]['m'] = m; 
        
        return ds

    
        
    
class RP20(GarrettMunkSpectralModel):
    """
    Rob pinkel internal wave spectral model
    """
    def __init__(self,N,smparams=GM81Params(),latitude=30,jstar=2,nmodes=10):
        super().__init__(smparams,N,latitude,jstar,nmodes=nmodes)
    
    def R(self,omega,j):
        threshold = 1/tocph # 1cph
        sig = 1 / (1 + np.exp(-100*(omega-threshold)*tocph))
        rolloff = (1-sig) + sig*np.exp(-(j-1)*(omega-threshold)*tocph)
        return rolloff
    
    def Comega(self,omega):
        return 1/np.sum([self.H(j)*self.R(omega,j) for j in range(1,self.nmodes+1)])
   

    #Internal Wave Displacement Power
    def FD(self,omega,alpha=None,mode=None,):
        Co  = [(self.Momega(o)) for o in omega]
        if alpha is not None:
            ktom = (self.b/np.pi)*np.sqrt( (self.No**2-omega**2)/(omega**2-self.f**2) )
            mode = np.pi*ktom * alpha 
            P = (1/ktom)*self.B(omega)*self.H(mode)*self.M
            
        elif mode is not None:
            P =  self.B(omega)* self.H(mode)*self.R(omega,mode)*Co
        
        else:
            P = np.zeros(len(omega))
            #DIY BV cutoff
            for i,om in enumerate(omega):
                if om <= self.No:
                   P[i] =  self.B(om) 
                else:
                   P[i] = self.B(om)*(1 - (om-self.No)/self.No)
            
        return self.A * ( (omega**2 - self.f**2)/omega**2 ) *P
    
    
        
        
