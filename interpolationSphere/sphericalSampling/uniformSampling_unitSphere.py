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
    p_euclide_2d:               Sample positions in euclidean coordinates
    p_sph:                      Sample positions in spherical coordinates

    Source:
    _______
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
    '''
    This function returns a spherical Fibonacci lattice. By using the golden ration
    we can define a Fibonacci lattice on the unit disc/square. The set of points on
    these surfaces, provide a global equal space packing. These points a projected on
    the sphere by using Lamberts cylindrical equal-area mapping.

    Parameters:
    ___________
    N:                          Number of sample positons

    return:
    _______
    p_euclide_3d:               Sample positions in euclidean coordinates
    p_sph:                      Sample positions in spherical coordinates

    Source:
    _______
    Point Sets on the Sphere S^2 with Small Spherical Cap Discrepancy,Johann S. B., 2011
    '''

    #Uniform distributed unit square:
    p_square_2d = fibonacci_lattice_unitSquare(N)
    alpha = p_square_2d[:,0]
    tau = p_square_2d[:,1]

    #Wrap around sphere using Lamberts cylindrical equal-area mapping
    x = 2*np.sqrt(tau-tau**2)*np.cos(2*np.pi*alpha)
    y = 2*np.sqrt(tau-tau**2)*np.sin(2*np.pi*alpha)
    z = 1 - 2*tau

    #Evenly distributed points on the unit sphere:
    #1) [x,y,z]:
    p_euclid_3d = np.column_stack((x,
                                y,
                                z))

    #2) [r,phi,theta]:
    p_sphere = utilities.euclid_2_sphere(p_euclid_3d)

    return p_euclid_3d, p_sphere


def fibonacci_lattice_unitSquare(N):
    '''
    This functions returns a unit square lattice constructed by using the
    golden ratio.

    Parameters:
    ___________
    N:                              Number of points within the lattice

    return:
    _______
    p_euclide_2d:                   Lattice coordinates in cartesian coordinates

    Source:
    _______
    '''
    n = np.linspace(0,N,N,endpoint=False)
    gr = (1+np.sqrt(5))/2                               #golden ratio (we need irrational number)

    #Fibonacci grid on the unit square: (x,y) in [0,1]^2
    x = n/gr%1
    y = n/N

    return np.column_stack((x,y))

def fibonacci_lattice_unitDisc(N):
    '''
    This functions returns a unit disc lattice constructed by using the
    golden ratio.

    Parameters:
    ___________
    N:                      Number of points within the lattice

    return:
    _______
    p_euclide_2d:           Lattice coordinates in cartesian coordinates
    p_pol:                  Lattice coordinates in polar coordinates

    Source:
    _______
    Fibonacci grids: A novel approach to global modelling, Richard S. et al., 2006
    '''

    n = np.linspace(0,N,N,endpoint=False)
    gr = (1+np.sqrt(5))/2                               #golden ratio (we need irrational number)

    phi = 2*np.pi*n/gr
    r = np.sqrt(n/N)

    p_pol = np.column_stack((r,phi))
    p_euclide_2d = utilities.polar_2_euclid(p_pol)

    return p_euclide_2d, p_pol


def spherical_cap_discrepancy(p):
    '''
    This function returns the spherical set discrepancy of the given point set p
    on the unit sphere. (p in S^2)

    Parameters:
    ___________
    p:                          Point set in S^2

    return:
    _______
    scd:                        Spherical cap discrepancy

    Source:
    _______
    '''

    #Arbitrary center points of caps on unit sphere
    W = 21
    w_theta = np.linspace(0,np.pi,W,endpoint=True)
    w_phi = np.linspace(0,2*np.pi,W,endpoint=True)
    w_r = [1]*W

    w_sphere = np.column_stack((w_r,
                                w_phi,
                                w_theta))

    w_euclid_3d = utilities.sphere_2_euclid(w_sphere)

    #Arbitrary distance measure of points
    T = 21
    t = np.linspace(-1,1,T,endpoint=True)

    #Points within cap specifications
    c_wt = [p @ w_euclid_3d.T > t]



#%%

W = 21
w_theta = np.linspace(0,np.pi,W,endpoint=True)
w_phi = np.linspace(0,2*np.pi,W,endpoint=True)
w_r = [1]*W

w_sphere = np.column_stack((w_r,
                                w_phi,
                                w_theta))

w_euclid_3d = utilities.sphere_2_euclid(w_sphere)

t = np.array([-1,-0.5,0,0.5,1])




sk = (w_euclid_3d[:2] @ w_euclid_3d.T).T[:,:,None]

mask = sk > t

sk = np.einsum('kli->kil', mask)

w_euclid_3d[:2][:,:,None][sk]





















#%%