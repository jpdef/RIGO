#!/usr/bin/python3
#RIGO
#Desc : Script that executes internal wave simulation 
#Auth : J. DeFilippis
#Date : 11-13-2024 
from configparser import ConfigParser
from argparse import ArgumentParser
from functools import partial
from tqdm import tqdm
import itertools
import os
import sys
import numpy as np

from Simulator import InternalWaveSimulator
from ModeSolver import InternalWaveModes
import SpectralDensityModel as SDM
tocph = (2*np.pi)/3600

def make_vec(tuple_string):
    """
    Takes a string tuple ('start_value','end_value','increment_value') and turns
    it into a 1-dimensional numpy vector
    """
    tuple_float  = tuple(float(x.strip()) for x in tuple_string.split(',')) 
    return  np.arange( *tuple_float )

def coriolis(latitude):
    #rotation rate of earth in rad/s
    omega = 7.292115*1e-5
    f = 2*omega*np.sin(np.pi*latitude/180)
    return f

def build_physical_axis(config):
    """
    This takes the physical coordinates specified as 1-dim vectors (x,y,z,t)
    and creates a 4-dim space with x,y,z,t as fields of that array.
    """
    x = make_vec( config.get('physical_axis','xaxis') )
    y = make_vec( config.get('physical_axis','yaxis') )
    z = make_vec( config.get('physical_axis','zaxis') )
    t = make_vec( config.get('physical_axis','taxis') )
    
    """
    Check for errors in the parameter space
        - x,y,z,t >= 0 
    """
    assert( all(x >= 0) & all(y >= 0) & all(z >= 0) & all(t >= 0) )

    
    D = combine_parameters([x,y,z,t],['x','y','z','t'],(float,float,float,float))
    return D


def build_parameter_space(config):
    """
    This takes the parameter coordinates 1-dim vectors (omega,mode numbers,thetas)
    and creates a 4-dim space with o,m,t as fields of that array.
    """
    omegas =  make_vec( config.get('parameter_space','omegas') )*tocph
    modes  =  make_vec( config.get('parameter_space','modes') )
    thetas =  make_vec( config.get('parameter_space','thetas') )


    """
    Check for errors in the parameter space
        - Mode numbers should be positive integer > 0 
        - Lowest frequency - df > f0 (inertial frequency)
    """
    f0 = coriolis(30)
    assert( all(modes > 0) )
    assert( (omegas[0] - np.diff(omegas)[0]) > f0 )   


    D = combine_parameters([omegas,modes,thetas],['o','m','t'],(float,int,float))
    return D
    

def combine_parameters(vecs,vecnames,types):
    """
    This takes a list of 1-dimension numpy arrays and names associated with those arrays
    and then turns this into a n-dimensional array with fields that have the vector names.
    """
    dtype = list(zip(vecnames,types))
    AX = np.array(list(itertools.product(*vecs)),dtype=dtype)
    AX = AX.reshape( tuple([ len(v) for v in vecs ]) )
    return AX


def add_horizontal_wavenumbers(parameter_space, vmodes):
    """
    The wave parameter space is 2-dimensional frequency, mode number (o,m). The horizontal
    wavenubmer, k, is calculated with the dispersion relationship (o,m) -> k. This is stored in 
    vmodes parameter. This function adds a horizontal wavenumber field to the parameter space 
    array and then takes the stored wavenumber values and assigns them to the array
    """
    parameter_space = utils.add_dtype( parameter_space, ('k',float) )
    parameter_space['k'] = 0.0
    for k,vmode in progressbar(vmodes,'Adding horizontal wavenumbers'):
        mm = len(vmode.hwavenumbers); tl = parameter_space.shape[2]
        parameter_space[k,:mm,:]['k'] = np.tile(vmode.hwavenumbers,(tl,1)).T
        parameter_space[k,mm:,:]['m'] = 0
    
    return parameter_space


def progressbar(dataset,desc):
    """
    Helper function that wraps the tqdm library to make 
    function call shorter
    """
    iterator = enumerate(dataset)
    return tqdm(iterator,ascii=True,total=len(dataset),leave=True,desc=desc)


if __name__ == "__main__":
    
    #Load Arguments
    parser = ArgumentParser(description="RIGO (Random Inertia Gravity Oscillations)")
    parser = ArgumentParser(prefix_chars='@')
    parser.add_argument('@@configfile',default='./config.ini') 
    args = parser.parse_args()
    
    #Load input configuration
    config = ConfigParser()
    config.read(args.configfile)
    
    
    #Build physical space
    physical_axis   = build_physical_axis(config)
    physical_axis   = utils.add_dtype( physical_axis, ('d',float) )
    
    #Build parameter space
    parameter_space =  build_parameter_space(config)
    
    #Solve for modes
    omegas = np.unique(parameter_space['o']); modes = np.unique(parameter_space['m'])
    stratification = utils.loadnpy(config.get('filepaths','stratification_file') )
    N2 = stratification[1]; Z = stratification[0];num_modes=int(modes[-1])
    internal_wave_modes = []
    for i,omega in progressbar(omegas, desc='Building modes ...'):
        internal_wave_modes.append( InternalWaveModes(N2,Z,omega,num_modes=num_modes)  )
    
    parameter_space =  add_horizontal_wavenumbers(parameter_space, internal_wave_modes)
     
    
    #Build spectrum model
    gm81 = SDM.GM81Params()
    gm81.No = 6*tocph
    gmsm = SDM.GarrettMunkSpectralModel(gm81)
    
    #Calculate the variance of each parameter
    energy = np.zeros(shape=parameter_space.shape) # ie <u^2> or <zeta^2>
    dmode = 1; domega = np.mean(np.diff(omegas))
    PSD = gmsm.VI_PSD_PE
    for i,o in progressbar( parameter_space[:,0,0]['o'],desc='Sampling Spectrum'):
        SX = gmsm.sample_variance_1d(PSD,o,domega)
        
        #Count the number of modes numbers in the parameter that are greater than zero
        mask  =   parameter_space[i,:,0]['m'] > 0 
        mm = mask.sum()
        
        #Iterate over modes that are not zero ( zero modes are evanscent )
        for j in range(mm):
            energy[i,j,:] = SX * gmsm.H(parameter_space[i,j,0]['m'],max_modes=mm)
    
    #Prune parameter space. Remove parameters from the parameter vector where energy from the spectrum is zero
    energy_vector = energy.flatten()
    ps_vector = parameter_space.flatten()
    zero_mask = energy_vector > 0
    
    #Run simulation
    iws = InternalWaveSimulator(physical_axis,ps_vector[zero_mask],
                                energy_vector[zero_mask], internal_wave_modes)
    
    ensemble_number = int(config.get('simulation','ensemble_number'))
    output_filepath = config.get('filepaths','output_filepath')
    
    #Write output to files
    utils.savenpy(output_filepath + '/' + 'parameters.npy',parameter_space )
    for k,p in progressbar(np.arange(0,ensemble_number,1),'Running Ensemble'):
        iws.run()
        utils.savenpy(output_filepath + '/' + ('iws_out_itr=%00d.npy' % k),iws.phys_ax)
        