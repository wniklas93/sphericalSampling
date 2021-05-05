import numpy as np
import scipy as sp

from interpolationSphere.sphericalSampling import utilities
################################################################################
def interpolation_sphericalHarmonics(pi, pm, sm):
    '''
    This functions returns the interpolat at the points of interest (pi) for given
    supporting positions. The interpolation is based on spherical harmonics.

    Parameters:
    ___________
    pi:                 Interpolation positions (Points of interest)
    pm:                 Positions of supporting points
    sm:                 Sample values of supporting points
    return:
    _______
    si:                 Interpolant at interpolation positions

    Source:
    _______
    '''

    #Calculate coefficients of spherical harmonics
    c_sh = _ieq_sphericalHarmonics(pm,sm)

def _ieq_sphericalHarmonics(pm,sm):
    '''
    This functions solves the equation system for the given supporting
    points, namely, it returns the spherical harmonics coefficients.

    Parameters:
    ___________
    pm:                 Positions of supporting points
    sm:                 Sample values of supporting points

    return:
    _______
    c_sh:               Coefficients of spherical harmonics

    Source:
    _______
    "Handbook of Geomathematics, Practical Considerations (p. 2594),
    Willi Freeden et al. 2015"
    '''
    #Number of supporting points
    M = len(sm)

    #Maximum eigenvalue of Legendre differential eq.:
    c0 = 343                    #speed of sound [m/s]
    fMax = 20000                #Maximum frequency [Hz]
    wMin = c0/fMax              #Minimum wavelength
    L = int(2*np.pi/wMin)+1     #Maximum eigenvalue
    L = 20
    #Elevation-Azimuth-Eignevalue combinations:
    Nc = (L+1)**2                                                           #Number of coefficients
    lm = np.zeros((Nc,2))
    l = np.linspace(0,L,L+1,endpoint=True,dtype=int)
    m = [np.linspace(-l_,l_,2*l_+1,endpoint=True, dtype=int) for l_ in l]
    for l_ in l:
        start = l_**2
        end = (l_+1)**2
        N = end - start
        lm[start:end,:] = np.column_stack(([l_]*N,m[l_]))

    #Spherical harmonics at supporting points:
    #shape(Y): M, lm
    Y = sp.special.sph_harm(lm[:,1], lm[:,0],pm[:,0,None],pm[:,1,None])

    #state equation system and get Legendre coefficients:
    #Since M is greater than the number of coefficients, our system is over-
    #determined. Thus, in the least square error sense, we have to use the
    #pseudo inverse:
    assert M >= Nc, '''Number of supporting points must be greater than
    number of spherical harmonics coefficients!'''

    pY = np.linalg.inv(Y.T @ Y) @ Y.T
    print(np.shape(pY))
    #c_sh = np.linalg.solve(pY,sm)

    return 0











# %%
