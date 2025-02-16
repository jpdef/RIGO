#Desc : Library for solving modes
#Auth : J. DeFilippis
#Date : 7-25-2020

import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import eig, eigh
from scipy.integrate import trapz, cumtrapz

class ModeSolver:
    """
    One dimensional eigen value solver in  the form of 
    LH v = - e D2v 
    @ domain (array) that defines the points to solve on
    @ left_matrix (array) matrix on the left hand side of the eq.
    @ boundary (tuple) values at enpoints of domain
    @ free_surface (bool) free surface boundary at top v' = v
    """
    def __init__(self,domain,left_matrix=None,
                 boundary=[0,0],free_surface=False):
        
        
        #Check that domain is regular
        if (max(np.diff( np.diff(domain) ) ) > 1e-5 ):
            print("Domain is not regular by 1e-5")
            raise
            
        if (left_matrix is None):
            left_matrix = -1*np.identity(len(domain))
    
        #Check that right matrix has correct dimensions
        elif (left_matrix.shape[0] != left_matrix.shape[1]):
            print("Right hand matrix not-square")
            raise
            
        elif (left_matrix.shape[0] != len(domain)):
            print("Right hand matrix dim != domain dim")
            raise
        
        self.domain = domain
        self.npts = len(domain) - 2
        self.delta = np.mean(np.diff(domain))
        self.LHM = left_matrix
        self.bounds = boundary
        self.free_surface = free_surface
        self.mode_funcs = None
        
        
    def solve(self,num_modes):
        if self.free_surface:
            D2 = self.centered_2nd_derivative_fs()
            LH = self.LHM[:-1,:-1]
            eigval,eigvec = eigh(LH,-D2)
            eigvec = np.vstack([eigvec,
                               self.bounds[1]*np.ones(self.npts+1)])
            
        else:
            D2 = self.centered_2nd_derivative()
            LH = self.LHM[1:-1,1:-1]
            intev = [self.npts-num_modes,self.npts-1]
            eigval,eigvec = eigh(LH,-D2,subset_by_index=intev)
            eigvec = np.vstack([self.bounds[0]*np.ones(num_modes),
                               eigvec,
                               self.bounds[1]*np.ones(num_modes)])
            
        ind = np.argsort(eigval)[::-1]
        return [eigval[ind],eigvec[:,ind]]
    
    
    def centered_2nd_derivative_fs(self):
        """
        Desc : Generates a tri-diagonal matrix for 2nd order 
               derivative using a finite difference, with a free surface boundary
        Returns :
                matrix : np matrix
        """
        N = self.npts + 1 
        D2 = np.zeros(shape=(N,N))
        for i in range(1,N-1):
            D2[i,i]   = -2 
            D2[i,i-1] = 1 
            D2[i,i+1] = 1 
        
        D2[0,1]  = -9.8*self.delta/2 
        D2[N-1,N-1] = -2
        D2[N-1,N-2] = 1
        
        D2 /= self.delta**2 
        return D2
        
        
    def centered_2nd_derivative(self):
        """
        Desc : Generates a tri-diagonal matrix for 2nd order 
               derivative using a finite difference
        Returns :
                matrix : np matrix
        """
        N = self.npts
        D2 = np.zeros(shape=(N,N))
        for i in range(1,N-1):
            D2[i,i]   = -2 
            D2[i,i-1] = 1 
            D2[i,i+1] = 1 
        
        D2[0,0]  = D2[N-1,N-1] = -2
        D2[0,1]  = D2[N-1,N-2] = 1
        
        D2 /= self.delta**2 
        return D2

    
    def centered_1st_derivative(self):
        """
        Desc : Generates a tri-diagonal matrix for 1st order 
               derivative using a finite difference
        Returns :
                matrix : np matrix
        """
        N = self.npts + 2
        D1 = np.zeros(shape=(N,N))
        for i in range(1,N-1):

            D1[i,i]   = 0 
            D1[i,i-1] = -1 
            D1[i,i+1] = 1 
        
        #One side boundary derivative
        D1[0,0]  = D1[N-1,N-1] = 0
        D1[0,1]     = 4
        D1[0,2]     = -1 
        D1[N-1,N-2] = -4
        D1[N-1,N-3] = 1
        
        D1 /= 2*self.delta 
        return D1
    
    
    def centered_1st_derivative_fs(self):
        """
        Desc : Generates a tri-diagonal matrix for 1st order 
               derivative using a finite difference with a free surface
        Returns :
                matrix : np matrix
        """
        N = self.npts + 2
        D1 = np.zeros(shape=(N,N))
        for i in range(1,N-1):
            D1[i,i]   = 0 
            D1[i,i-1] = -1 
            D1[i,i+1] = 1 
        
        #One side boundary derivative
        D1[N-1,N-1] = 0
        D1[N-1,N-2] = -4
        D1[N-1,N-3] = 1
        
        D1 /= 2*self.delta 
        D1[0,0]  = -1/self.delta
        return D1
        
    
    def evaluate_mode(self,mode_number,z):
        #print('evaluate_mode :: mode_number %d' % mode_number)
        if self.mode_funcs:
            return self.mode_funcs[mode_number-1](z)
            
        else:
            self.interpolate_modes()
            return  self.mode_funcs[mode_number-1](z)
    
    
    def interpolate_modes(self):
        self.mode_funcs = []
        for n in range(self.modes.shape[1]):
            fun = interp1d(self.domain,self.modes[:,n])
            self.mode_funcs.append(fun)
    
    
class InternalWaveModes(ModeSolver):
    def __init__(self,N2,Z,frequency,
                           latitude=30,
                           num_modes=100,
                           scalar_gradient=None,
                           free_surface=False,
                           puvmodes=False):
        
        self.fo = coriolis(latitude)
        self.frequency = frequency
        self.N2 = N2; self.Z = Z
        
        if ( self.frequency < self.fo ):
            print(frequency,self.fo)
            print("Frequency needs to be greater than coriolios")
            raise
        
        elif( self.frequency > np.sqrt(max(N2)) ):
            print(frequency,np.sqrt(max(N2)))
            print("Frequency cannot exceed max stratfication")
            raise
        
        F = frequency**2 * np.ones(len(Z))    
        LH = np.diag(N2-F)
        super().__init__(Z,LH,boundary=[0,0],free_surface=free_surface)
        
        self.hwavenumbers, self.modes = self.solve(frequency,num_modes=num_modes)
        self.normalize()
        
        if puvmodes:
            if free_surface:
                D1 = self.centered_1st_derivative_fs()
            else:
                D1 = self.centered_1st_derivative()
                
            self.modes = D1 @ self.modes 
        
        if scalar_gradient is not None:
            #Multiple rows by each element in gradient
            self.modes = (self.modes.T * scalar_gradient).T
    
    
    def solve(self,frequency,num_modes,phase_speed_cutoff=0.001):
        """
        Solves the matrix equation for mode shapes (eigenvectors) 
        and the phase speeds ( eigenvalues )
        @num_modes          : number of modes to calculate
        @phase_speed_cutoff : upper limit for the eigenvlaues to incorporate.
                              is the phase speed is too slow likely isn't physical
                              this is equivalent to a vertical wavenumber cutoff
        """
        eigvals, modes = super().solve(num_modes)
        frequency = self.frequency if frequency is None else frequency
        
        #Check that eigenvalues are real and positive
        pdmask = (~np.iscomplex(eigvals)) &  (eigvals.real > 0 ) 
        eigvals = eigvals[ pdmask ]
        
        #Check that eigenvalues are above phase speed cutoff
        psmask = (np.sqrt(eigvals) > phase_speed_cutoff)
        eigvals = eigvals[psmask]
        
        hwavenumber = np.sqrt( (frequency**2- self.fo**2) / eigvals)
        modes = modes[:,pdmask]
        modes = modes[:,psmask]
        return hwavenumber,modes
    
    
    def normalize(self):
        """
        Integrate PE = N^2 \psi^2  = 1
        """
        for j in range(self.modes.shape[1]):
            I = trapz(self.N2*self.modes[:,j]**2,self.Z)
            self.modes[:,j] /= np.sqrt(I)
        
    def hwavenumber(self,mode_number):
        return self.hwavenumbers[mode_number].real
    
    
    def wkbj_solve(self,mode_cutoff=50,Nob=6.8):
        """
        Desc : Solves helmholtz equation using WKB
               approximation
        Params :
           mode_cutoff : int
              number of modes to calculate 1..mode_cutoff
        Returns:
               solution : array
                 A solution as a function of the depth coordinate

        """
        
        
        #Stretch coordinate eta = 1/(No *B) int_{zbot}^z N(z') dz'
        eta = np.pad( cumtrapz( np.sqrt(self.N2), self.Z ), (1,0) )
        #WKB scaling factors
        Nob = eta[-1]
        sf  = np.sqrt(self.N2[0]/self.N2) 
        
        modes = np.zeros(shape=(len(self.Z),mode_cutoff))
        hwavenumbers = np.zeros(mode_cutoff)
        zn = np.arange(0,len(self.Z),1)
        for j in range(1,mode_cutoff):
            modes[:,j-1] = sf*np.sin((1/Nob)*np.pi*j*eta)
            hwavenumbers[j-1] = (np.pi*j*np.sqrt(self.frequency - self.fo))/Nob
        
        return hwavenumbers, modes
        

class QuasiGeostrophicModes(ModeSolver):
    def __init__(self,N2,Z,frequency=0,scalar_gradient=None,free_surface=False,latitude=30):
        LH = np.diag(N2)
        self.fo = coriolis(latitude)
        self.frequency=frequency
        super().__init__(Z,LH,free_surface=free_surface)

        self.hwavenumbers, self.modes = self.solve()
        
        if scalar_gradient is not None:
            #Multiple rows by each element in gradient
            self.modes = (self.modes.T * scalar_gradient).T
    
    
    def solve(self):
        eigval, eigvec = super().solve()
        eigval = self.fo/np.sqrt(eigval)
        return eigval,eigvec

    
    def hwavenumber(self,mode_number):
        return self.hwavenumbers[mode_number]
    
    
def coriolis(latitude):
    #rotation rate of earth in rad/s
    omega = 7.292115*1e-5
    f = 2*omega*np.sin(np.pi*latitude/180)
    return f
