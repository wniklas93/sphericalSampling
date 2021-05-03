import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
################################################################################
def euclid_2_sphere(p):
    '''
    This function transforms the euclidean representation of a point in R^3 to
    the spherical representation. This function accepts single points and
    multiple points. The format of the input must be as follows:
    p = [x,y,z]

    Parameters:
    ___________
    p:                      Points of R^3 in euclidean representation

    return:
    _______
    p_sphere:              Points of R^3 in spherical representation
    '''

    p = np.atleast_2d(p)

    x = p[:,0]
    y = p[:,1]
    z = p[:,2]

    #Transform: [x,y,z] --> [r,phi,theta]
    p_sphere = np.column_stack((np.sqrt(x**2 + y**2 + z**2),             #r
                                np.arctan2(y,x),                          #phi
                                np.arctan2(np.sqrt(x**2 + y**2),z)))      #theta

    return p_sphere

def sphere_2_euclid(p):
    '''
    This function transforms the spherical representation of a point in R^3 to
    the euclidean representation. This function accecpts single points and
    multiple points. The format of the input must be as follows:

    p = [r,phi,theta],
    where r in (-inf, inf), phi in [0,2*pi], theta in [0,pi]

    Parameters:
    ___________
    p:                          Points of R^3 in spherical representation

    return:
    _______
    p_euclidean:                Points of R^3 in euclidean representation
    '''


    p = np.atleast_2d(p)

    r =         p[:,0]
    phi =       p[:,1]
    theta =     p[:,2]

    #Transform: [r,phi,theta] --> [x,y,z]
    p_euclid_3d = np.column_stack((r*np.cos(phi)*np.sin(theta),
                                  r*np.sin(phi)*np.sin(theta),
                                  r*np.cos(theta)))

    return p_euclid_3d


def polar_2_euclid(p):
    '''
    This function transforms points in polar coordinates representation to
    points in euclidean coordinates representation

    Parameters:
    ___________
    p:                  Points of R^2 in polar representation

    return:
    _______
    p_euclid_d:         Points of R^2 in euclidean representation
    '''
    p = np.atleast_2d(p)

    r = p[:,0]
    phi = p[:,1]

    x = r * np.cos(phi)
    y = r * np.sin(phi)

    p_euclid_2d = np.column_stack((x,y))
    return p_euclid_2d

def plot_3D(p, title):
    '''
    This functions plots the given points in the 3D space. The given
    points are elements of R^3.

    Parameters:
    ___________
    p:                              To be plotted points
    title:                          Title of plot


    return:
    _______
    -

    '''
    #Set colours and render
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(p[:,0], p[:,1], p[:,2], color='b', s=20)

    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.set_xlabel("x-Axis")
    ax.set_ylabel("y-Axis")
    ax.set_zlabel("z-Axis")
    ax.title.set_text(title)
    plt.tight_layout()
    plt.show()


def plot_2D(p, title):
    '''
    This functions plots the given points in the 2D space. The given
    points are elements of R^2.

    Parameters:
    ___________
    p:                              To be plotted points
    title:                          Title of plot


    return:
    _______
    -

    '''
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    ax.scatter(p[:,0], p[:,1], color='b', s=20)

    ax.set_xlabel("x-Axis")
    ax.set_ylabel("y-Axis")
    ax.title.set_text(title)
    plt.tight_layout()
    plt.show()







#%%
