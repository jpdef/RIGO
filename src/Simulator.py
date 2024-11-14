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
import iw_misc
    
def progressbar(self,dataset,desc):
    """
    Desc:
    Helper function that wraps the tqdm library to make 
    function call shorter
    """
    iterator = enumerate(dataset)
    return tqdm(iterator,ascii=True,total=len(dataset),leave=True,desc=desc)

            
class InternalWaveSimulator:
    def __init__(self,physical_axis, parameters, stratification, dsdm):
        """
        @ physical_axis   : random wave field is evaluated on these coordinates (x,y,z,t)
        @ parameters      : wave parameter space (omega,mode,variance)
        @ stratification  : stratification profile (n2,z) 
        @ energy          : variance of the internal wave amplitudes
        """
        self.parameters = parameters
        self.phys_ax = physical_axis 
        
        self.DSDM = dsdm
        self.N2 = stratification[1]; self.Z  = stratification[0];
        self.IWM = lm.InternalWaveModel(parameters,self.N2,self.Z)
        
        
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
        defined by D
        """
        
        chunk_size = int(1e5)
        pav_len = len(self.phys_ax.flatten())
        out = np.zeros(pav_len)
        for i in tqdm(np.arange(0,pav_len,chunk_size)):
            j = i + chunk_size
            out[i:j] = self.IWM.model_matrix(self.phys_ax.flatten()[i:j]) @ amp_vec.T
        
        self.phys_ax[field] =  out.reshape(self.phys_ax.shape) 
    
    
    def run(self):
        self.amplitudes = self.generate_amplitudes()
        amp_vec = np.concatenate([ self.amplitudes[0,:],self.amplitudes[1,:] ])
        self.run_forward(amp_vec)
        

    

    