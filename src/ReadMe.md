# Overiew
This code package is meant to help the user calculate modal ocean inertia-gravity waves. File ModeSolver contains code that calculates internal wave displacement and horizontal velocity modes given an varying stratification profile with depth. This package also allows the user to simulate random inertia-gravity waves with dynamic mode shapes over a 4-dimensional space. The main script rigo.py computes inertia-gravity fluctuations using a configuration file provided by the user. File SpectralDensity.py contains code to calculate the Garrett-Munk spectral curves which is fed into the simulation to get realistic inertia-gravity amplitude statistics. The simulation is designed to run multiple ensembles for statistical calulations using random inertia-gravity fields.

# Main script
The rigo script is designed as a command line tool that is fed a single configuration file that specifies the location of the input data and output data. 
```
rigo @@configfile config.ini
```
The output of each simulation contains a random displacement field evaluate at the grid points specified by the user. The parameter space is also saved in the output data. The most resource intensive computation is calculating the modes, the modes are frequency dependent and so computation time is O(number of modes x number of frequencies). Once the modes are calulated a series of matrix multiplications are done to calculate the wave field at each specified point. This can be memory intensive but if it requires too much memory the process is serialized, chunking the matrix multiplications.

# Configuration File 
The configuration file requires the user to specify a path to a stratification profile (N^2, Z) as well as a path where output can be placed. Note that N^2 should be specified in [rad/s]^2 and Z in meters. Each new simulation generates a random realization of wave phase and amplitude. To generate an ensemble of N random realization set the ensemble_number parameter. An output data file is produced for each ensemble member.
 ```
[filepaths]
output_filepath : ../data/sim
stratification_file : ../data/mystratification.npy

[simulation]
ensemble_number : 3
 ```
The physical space (x,y,z,t) that the internal wave field need to be evaluated on is also specified in the configuration files. Currently this space is specified as vectors with uniform spacing formatted as start, end, spacing. All spatial axes (x,y,z axis) are in meters and the time axis (taxis) is in seconds.
```
[physical_axis]
xaxis : 0.0, 1 , 1
yaxis : 0.0, 1 , 1
zaxis : 10.0, 500.0, 1
taxis : 0.0, 86400, 300
```
The parameter space ($\omega$,m,$\theta$) defines the basis functions which are modal waves $\phi_m(z)e^{k \dot x - \omega t}$. The frequency axis should be defined by the inertial frequency to the buoyancy frequency, the simulation discretizes the Garrett-Munk spectrum and so the accuracy of the simulation increases with smaller frequency sampling. The number of modes determines the accuracy of the vertical structure of the simulation, there is a mode limit (which is frequency dependent) that is calculated internally, in general 50 modes is sufficient. The theta parameter can be used to specify the wave direction for three-dimensional problems, in general the wave spectrum is assumed to be horiztonally isotropic. For two-dimensional problems this can be set to 0 like in the example below.
```
[parameter_space]
omegas : 0.05, 10.0, 0.1
modes  : 1 , 50, 1
thetas : 0,1,1
```

# Other uses
A example jupyter notebook is provided in the notebooks directory. This gives examples on to use parts of the rigo library to calculating internal wave modes, generating dispersion curves, and Garrett-Munk spectra curves. These are useful for various oceanographic problems involving internal waves as well as diagnosing the simulation results. 