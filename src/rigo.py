#!/usr/bin/python3
#RIGO
#Desc : Script that executes internal wave simulation 
#Auth : J. DeFilippis
#Date : 11-13-2024 
tocph = 3600/(2*np.pi)
from configparser import ConfigParser
from argparse import ArgumentParser
import itertools
import sys

from Simulator import InternalWaveSimulator
import SpectralDensityModel as SDM
tocph = (2*np.pi)/3600

def build_physical_axis(config):
    x = make_vec( config.get('physical_axis','xaxis') )
    y = make_vec( config.get('physical_axis','yaxis') )
    z = make_vec( config.get('physical_axis','zaxis') )
    t = make_vec( config.get('physical_axis','taxis') )
    
    D = combine_parameters([x,y,z,t],['x','y','z','t'],(float,float,float,float))
    return D

def build_parameter_space(config):
    omegas =  make_vec( config.get('parameter_space','omegas') )*tocph
    modes  =  make_vec( config.get('parameter_space','modes') )
    thetas =  make_vec( config.get('parameter_space','thetas') )
    
    D = combine_parameters([omegas,modes,thetas],['o','m','t'],(float,int,float))
    return D
    
def make_vec(tuple_string):
    tuple_float  = tuple(float(x.strip()) for x in tuple_string.split(',')) 
    return  np.arange( *tuple_float )

def combine_parameters(vecs,vecnames,types):
    dtype = list(zip(vecnames,types))
    AX = np.array(list(itertools.product(*vecs)),dtype=dtype)
    AX = AX.reshape( tuple([ len(v) for v in vecs ]) )
    return AX

if __name__ == "__main__":
    
    #Load Arguments
    parser = ArgumentParser(description="RIGO (Random Inertia Gravity Oscillations)")
    parser = ArgumentParser(prefix_chars='@')
    parser.add_argument('@@configfile',default='./config.ini') 
    args = parser.parse_args()
    
    #Load input configuration
    config = ConfigParser()
    config.read(args.configfile)
    
    
    #Build physical space, parameter space
    physical_axis   = build_physical_axis(config)
    parameter_space = build_parameter_space(config)
    stratification = utils.loadnpy(config.get('filepaths','stratification_file') )
    
    physical_axis   = utils.add_dtype( physical_axis, ('d',float) )
    #parameter_space = utils.add_dtype( parameter_space, ('E',float) )
    
    #Build spectrum model
    gm81 = SDM.GM81Params()
    gm81.No = 6*tocph
    gmsm = SDM.GarrettMunkSpectralModel(gm81)
    energy = gmsm.generate_discrete_spectrum_2d(parameter_space['o'][:,0,0],
                                                parameter_space['m'][0,:,0])['E']
    
    #Run simulation
    iws = InternalWaveSimulator(physical_axis,parameter_space.flatten(),stratification,energy)
    iws.run()
    
    #Write output to file
    #utils.savenpy(iws.phys_ax,outfiles['dataspace'])
    #utils.savenpy(iws.wps,outfiles['parameterspace'])