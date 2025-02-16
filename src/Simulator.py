#!/usr/bin/python3
#SIMULATE
#Desc : Script that executes internal wave simulation 
#Auth : J. DeFilippis
#Date : 7-16-2019

import sys
import numpy as np
import pandas as pd
import feather
import json
import os 
import functools
import scipy
import scipy.interpolate as interp
import itertools
from functools import partial
from tqdm import tqdm
 
#Source local libraries
sys.path.append('../src/lin_model')
sys.path.append('../src/misc')

import LinearModel as lm

def progressbar(dataset,desc):
    """
    Desc:
    Helper function that wraps the tqdm library to make 
    function call shorter
    """
    iterator = enumerate(dataset)
    return tqdm(iterator,ascii=True,total=len(dataset),leave=True,desc=desc)
    
            
class InternalWaveSimulator:
    def __init__(self,physical_axis, parameters, energy, internal_wave_modes):
        """
        @ physical_axis        : random wave field is evaluated on these coordinates (x,y,z,t)
        @ parameters           : wave parameter space (omega,mode,variance)
        @ internal_wave_modes  : list of internal wave modes
        @ energy               : variance of the internal wave amplitudes
        """
        self.parameters = parameters
        self.phys_ax = physical_axis 
        self.internal_wave_modes = internal_wave_modes
        self.DSDM = energy
        
        #Linear Model Objects (allow for constructing design matrices on the fly)
        #   PWLM = Plane wave linear model
        #   MDLM = Modal linear model
        self.PWLM = lm.LinearModel([self.planar_sine,self.planar_cosine],
                                   [self.parameters]*2)
        self.MDLM = lm.LinearModel([self.mode_function],[self.parameters])
        
        #Build Mode and Plane Wave Matrices
        self.PHI = self.MDLM.model_matrix(self.phys_ax[0,0,:,0].flatten())
        self.PSI = self.PWLM.model_matrix(self.phys_ax[:,:,0,:].flatten())
        
        
    def generate_amplitudes(self):
        """
        Generate gaussian random amplitudes given the variance at each wave parameter.
        Two amplitudes are generated corresponding to real/imag components.
        """
        dsdm = self.DSDM.flatten()
        amps = np.zeros(shape=(2,len(dsdm)))
        for i,p in enumerate(self.parameters.flatten()):
            v = np.random.normal(loc=0,scale=np.sqrt(dsdm[i]),size=2) 
            amps[0,i] = v[0]; amps[1,i] = v[1]
        
        return amps
    
    def run_forward(self,amp_vec,field='d'):
        """
        Desc:
        Runs foward problem with internal wave model and axis (x,y,z,t)
        Under normal conditions the number of parameters times the number of 
        grid points exceeds the memory so we need to chunk the data 
        num(o,m,theta) * num(x,y,z,t) > memory limit
        
        Assumption is that mode functions are the most expensive computation
        so we first create a matrix PHI(z; o,m) that has all mode funcitons
        evaluated at every depth grid point.
        PHI is (num(z) x num(omega * mode))
        
        Then we create plane wave matrix PSI (x,y,t ; o, k(m), theta), this is 
        where we chunk the data in groups of (x,y,t) and compute those groups
        of coordinates in a for loop. At the end of the for loop we create an
        output vector Zeta(x,y,z,t) = ( PSI . PHI ) @ A
        
        """
        
        itr_coords = self.phys_ax[:,:,0,:].flatten()
        ind = lambda x,xv : np.argmin( abs(x -  xv))
        
        #Multiply with Planar Waves
        for i,itr_coord in progressbar(itr_coords,'Evaluating field...'):
            zeta = self.PSI[i,:] * np.hstack([self.PHI,self.PHI]) @ amp_vec.T
           
            ci = tuple([ ind(itr_coord['x'],self.phys_ax['x'][:,0,0,0]),
                         ind(itr_coord['y'],self.phys_ax['y'][0,:,0,0]),
                         slice(0,None,1),
                         ind(itr_coord['t'],self.phys_ax['t'][0,0,0,:]) ])
           
            self.phys_ax[ci][field] = zeta
        
        
    
    def run(self):
        self.amplitudes = self.generate_amplitudes()
        amp_vec = np.concatenate([ self.amplitudes[0,:],self.amplitudes[1,:] ])
        self.run_forward(amp_vec)
        
  
    """
    These functions define the basis functions of the linear model object. The horizontal dimensions
    have planar waves which is a combination of the planar cosine and sine. The vertical dimensions use the 
    modal_function to fill the space.
    """

    def planar_cosine(self,o,m,t,K):
        k =  K*np.cos(t*np.pi/180)
        l =  K*np.sin(t*np.pi/180)
        
        def function(args):
            arg = 2*np.pi*(k*args['x'] + l*args['y']- o*args['t'])
            ret = np.cos(arg)
            return ret.real
        
        return function
    
    def planar_sine(self,o,m,t,K):
        k =  K*np.cos(t*np.pi/180)
        l =  K*np.sin(t*np.pi/180)
        
        def function(args):
            arg = 2*np.pi*(k*args['x'] + l*args['y']- o*args['t'])
            ret = np.sin(arg)
            return ret.real
        
        return function

    def mode_function(self,o,m,t,K):
        freqs = np.array( [mode.frequency for mode in self.internal_wave_modes] )
        i = np.argmin(abs( o - freqs ) )
        
        def function(args):
            ret = self.internal_wave_modes[i].evaluate_mode(m,args['z'])
            return ret.real
        
        return function