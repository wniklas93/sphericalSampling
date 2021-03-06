import numpy as np
import scipy as sp

from sphericalSampling import resamplingOnSphere
from sphericalSampling import uniformSampling_unitSphere

################################################################################
def test_k_h():
    #Check on positive definitness of kernel matrices generated by kernel
    #functions. If kernel matrices are positive definite then kernel function
    #is strictly positive definite:

    #1) generate sample points:
    N = 500                     #Number of points
    p_euclid_f, p_sphere_f = uniformSampling_unitSphere.sampleUnitSphere_geometric_fibonacci(N)

    #2)Get kernel functions
    k_f_gaussian = resamplingOnSphere.kernel('gaussian')
    k_f_invMultiquadric = resamplingOnSphere.kernel('invMultiquadric')
    k_f_tpf = resamplingOnSphere.kernel('TPF', s=3)

    #2) generate kernel matrices for different kernel functions:
    k_m_gaussian = resamplingOnSphere._k_h(k_f_gaussian, p_euclid_f, p_euclid_f)
    k_m_invMultiquadric = resamplingOnSphere._k_h(k_f_invMultiquadric, p_euclid_f, p_euclid_f)
    k_m_tpf = resamplingOnSphere._k_h(k_f_tpf, p_euclid_f, p_euclid_f)

    #3) Check on symmetry:
    assert np.all(k_m_gaussian == k_m_gaussian.T)
    assert np.all(k_m_invMultiquadric == k_m_invMultiquadric.T)
    assert np.all(k_m_tpf == k_m_tpf.T)

    #4)Check on positive definitness:
    eps = 0.00001

    u, v = np.linalg.eig(k_m_gaussian)
    assert np.all(np.abs(np.imag(u)) <= eps)
    assert np.all(np.real(u) > -eps)


    u, v = np.linalg.eig(k_m_invMultiquadric)
    assert np.all(np.abs(np.imag(u)) <= eps)
    assert np.all(np.real(u) > -eps)

    u, v = np.linalg.eig(k_m_tpf)
    assert np.all(np.abs(np.imag(u)) <= eps)
    assert np.all(np.real(u) > -eps)


# def test_sequence_An():
#     #Todo
#     #The Sobolev space H(An,W) is parametrized by the sequence An and the W,
#     #where W denotes the space the function being element of H(An,W) are de-
#     #fined on. For n=inf An must diverge against inf. If An=inf, then the
#     #Sobolev norm is a non-smoothness measure


#     #1)Get kernel functions
#     k_f_gaussian = interpolationOnSphere.kernel('gaussian')
#     k_f_invMultiquadric = interpolationOnSphere.kernel('invMultiquadric')
#     k_f_tpf = interpolationOnSphere.kernel('TPF', s=3)

#     #2)Get sequences
#     An_gaussian = interpolationOnSphere.sequence_An(k_f_gaussian)
#     An_invMultiquadric = interpolationOnSphere.sequence_An(k_f_invMultiquadric)
#     An_tpf = interpolationOnSphere.sequence_An(k_f_tpf)


#%%

















#%%
