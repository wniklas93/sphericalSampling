import numpy as np
import scipy as sp

from interpolationSphere.sphericalSampling import utilities
################################################################################
def interpolation_sphericalHarmonics(pi, pm, sm):
    '''
    This functions returns the interpolant at the points of interest (pi) for given
    supporting positions. The interpolation is based on spherical harmonics.
    Note:
    -   Currently, spherical harmonics up to the 30th degree (Eigenvalues) are used
        for the linear combination (Source: Loudness stability of binaural ...).

    -   Interpolation and supporting point positions must be distributed over the
        unit sphere. Hence, these positions are represented by phi and theta
        coordinates: [[phi0, theta0],
                      [phi1, theta1],...]

    -  The to be approximated function must be real valued.

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
    -   Handbook of Geomathematics, Practical Considerations (p. 2594),
        Willi Freeden et al. 2015

    -   Loudness stability of binaural soundwith spherical harmonic representation
        of sparse head-related transfer functions, Zamir Ben-Hur et al., 2019
    '''

    assert pi.ndim == 2, "Interpolation point positions must be 2 dimensional!"
    assert pm.ndim == 2, "Supporting point positions must be 2 dimensional!"
    assert sm.ndim == 1, "Supporting point sample values must be 1 dimensional!"
    assert np.all(np.imag(sm) == 0), "Supporting point sample values must be real valued!"

    #Maximum eigenvalue of Legendre differential eq.:
    L = 40

    #Calculate coefficients of spherical harmonics
    c_sh = _coeffs_sphericalHarmonics(pm,sm,L)

    #Determine linear combination of spherical harmonics:
    lm = _sh_degreeOrder_combinations(L)
    Yi = sp.special.sph_harm(lm[:,1], lm[:,0],pi[:,0,None],pi[:,1,None])

    si = Yi @ c_sh

    assert np.any(np.abs(np.imag(si)) < 0.00001), '''Interpolant must be real valued as the
    underlying supporting point sample values are real valued!'''

    si = np.real(si)

    return si

def max_squaredError(s_ref, si):
    '''
    Returns the maximum squared error between interpolant and reference.

    Parameters:
    __________
    s_ref:                      Reference sample values
    si:                         Sample values of the interpolant

    return:
    _______
    err:                        Maximum squared error

    Source:
    _______
    '''

    return np.amax(s_ref - si)**2

def _sh_degreeOrder_combinations(L):
    '''
    This functions returns the degree-order combinations in terms of
    spherical harmonics for the given maximum degree.

    Parameters:
    ___________
    L:                      Maximum degree

    return:
    _______
    lm:                     Degree-order combinations

    Source:
    _______

    '''
    #Number of coefficients:
    Nc = (L+1)**2

    lm = np.zeros((Nc,2))
    l = np.linspace(0,L,L+1,endpoint=True,dtype=int)
    m = [np.linspace(-l_,l_,2*l_+1,endpoint=True, dtype=int) for l_ in l]
    for l_ in l:
        start = l_**2
        end = (l_+1)**2
        N = end - start
        lm[start:end,:] = np.column_stack(([l_]*N,m[l_]))

    return lm

def _coeffs_sphericalHarmonics(pm,sm,L):
    '''
    This functions solves the equation system for the given supporting
    points, namely, it returns the spherical harmonics coefficients.

    Parameters:
    ___________
    pm:                 Positions of supporting points
    sm:                 Sample values of supporting points
    L:                  Maximum degree of spherical harmonics

    return:
    _______
    c_sh:               Coefficients of spherical harmonics

    Source:
    _______
    -   Handbook of Geomathematics, Practical Considerations (p. 2594),
        Willi Freeden et al. 2015

    '''
    #Number of supporting points
    M = len(sm)

    #Number of coefficients:
    Nc = (L+1)**2

    #Elevation-Azimuth-Eignevalue permutations:
    lm = _sh_degreeOrder_combinations(L)

    #Spherical harmonics at supporting points:
    #shape(Y): M, lm
    Y = sp.special.sph_harm(lm[:,1], lm[:,0],pm[:,0,None],pm[:,1,None])

    #state equation system and get coefficients:
    #Since M is greater than the number of coefficients, our system is over-
    #determined. Thus, in the least square error sense, we have to use the
    #pseudo inverse:
    assert M >= Nc, '''Number of supporting points must be greater than
    number of spherical harmonics coefficients!'''

    pY = np.linalg.inv(Y.T @ Y) @ Y.T

    c_sh = pY @ sm

    return c_sh











# %%
