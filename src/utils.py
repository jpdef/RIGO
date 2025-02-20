import datetime as dt
import numpy as np 
from tqdm import tqdm
import itertools
import pickle 
import netCDF4

"""
Numpy helpers  
"""
def add_dtype(X,new_type):
    if new_type in X.dtype.fields:
        return X
    new_dtype = np.dtype ( X.dtype.descr + [new_type] )
    Xnew = np.empty(shape=(X.shape),dtype=new_dtype)
    for field in X.dtype.fields:
        Xnew[field] = X[field]
    return Xnew


def mask_hi_lo(X,lower_bound,upper_bound):
    #for when I forget the order
    if upper_bound < lower_bound:
       tmp = upper_bound
       upper_bound = lower_bound
       lower_bound = tmp

    mask = (  (X > lower_bound) & (X < upper_bound) )
    return mask

def ind(x,xv):
    return np.argmin( abs(x-xv))

def find_date_index(X,timestamp):
    datetime = np.datetime64(timestamp)
    ii = np.argmin( abs( X - datetime) )
    return ii


def get_slice_indices(X,start_value,end_value,sampling=1):
    ts = np.nanargmin( abs( X - start_value) )
    te = np.nanargmin( abs( X - end_value)  )
    return slice(ts,te,sampling)

def dcheck(V):
    """Quickly prints useful stats for the data"""
    print("Mean %.2f, Min %.2f, Max %.2f," % 
          (np.nanmean(V),np.nanmin(V),np.nanmax(V) ) )
    print("Has Nan's : ", np.any(np.isnan(V)) )


def loadnpy(filename):
    with open(filename,'rb') as f:
        D = np.load(f)
    
    return D

def savenpy(filename,D):
    with open(filename,'wb') as f:
        np.save(f,D)
    
def loadpkl(filename):
    with open(filename,'rb') as f:
        D = pickle.load(f)
    
    return D


def savepkl(filename,D):
    with open(filename,'wb') as f:
        pickle.dump(D,f)
    

def save_ndarray_to_netcdf(array, filename, variable_names, dimensions, units=None):
    """
    Saves a NumPy array to a NetCDF file.

    Args:
        array (numpy.ndarray): The NumPy array to save.
        filename (str): The name of the NetCDF file to create.
        variable_name (str): The name of the variable in the NetCDF file.
        dimensions (tuple): A tuple of dimension names (strings).  Must match the 
                           number of array dimensions.  E.g., ('lat', 'lon') for a 2D array.
        units (str, optional): The units of the variable. Defaults to None.
    """

    with netCDF4.Dataset(filename, 'w', format='NETCDF4') as nc_file:  # Use NETCDF4 format for compression

        # Create dimensions
        for i, dim_name in enumerate(dimensions):
          nc_file.createDimension(dim_name, array.shape[i]) # Length of dim from array shape

        # Create variable
        var = nc_file.createVariable(variable_name, array.dtype, dimensions)
        if units:
            var.units = units

        # Write data
        var[:] = array  # Efficiently write the entire array