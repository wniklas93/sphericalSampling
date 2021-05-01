import numpy as np
import scipy as sp

from interpolationSphere.sphericalSampling import utilities
################################################################################
##########################Uniform Sampling of Sphere (S**2)#####################
def sampleUnitSphere_statistical_normal(N):
    '''
    This function returns the sampling positions of a sphere which has been
    resampled uniformly. The sampling approach is based on normal distributions:
    X~N(0,I_n), where X is a statistical independent variable with a normal distri-
    bution as probability function. Adding 3 random variables with normal distribution
    leads to a variable with normal distribution.

    Parameters:
    ___________
    N:                          Number of sample positons

    return:
    _______
    p_xyz:                      Sample positions in euclidean coordinates
    p_sph:                      Sample positions in spherical coordinates
    '''
    #create 3 standard normals
    Xx, Yx, Zx = np.random.normal(loc=0.0, scale=1, size=(3,N))
    l = np.sqrt(Xx**2 + Yx**2 + Zx**2)

    #Add those 3 random variables -> Since all points lie on the sphere they got
    #the same probability:
    #1) [x,y,z]:
    p_euclid_3d = np.column_stack((Xx/l, Yx/l, Zx/l))

    #2) [r,phi,theta]:
    p_sphere = utilities.euclid_2_sphere(p_euclid_3d)

    return p_euclid_3d, p_sphere

def sampleUnitSphere_geometric_fibonacci(N):
    #Evenly distributed points on unit square (0,1]² being element of R^3:
    n = np.linspace(0,N,N,endpoint=False)
    phi = (1+np.sqrt(5))/2
    p_euclid_3d = np.column_stack((n/phi%1,             #x
                                   n/N,                 #y
                                   [0]*N))              #z

    #Map unit square to unit sphere:
    #1) [r,phi,theta]:
    #p_sphere = np.column_stack()

    #2) [x,y,z]:
    p_euclid_3d = utilities.sphere_2_euclid(p_sphere)

    return p_euclid_3d, p_sphere
