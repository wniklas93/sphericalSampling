import numpy as np
import scipy as sp

from sphericalSampling import utilities
################################################################################
def test_coordinateTransforms():

    #Single point in euclidean representation
    p_euclid = [1,2,3]
    p_sphere = utilities.euclid_2_sphere(p_euclid)
    assert np.allclose(p_euclid,utilities.sphere_2_euclid(p_sphere))

    #Multiple points:
    p_euclid = [[1,2,3],
                [4,5,6],
                [7,8,9]]
    p_sphere = utilities.euclid_2_sphere(p_euclid)
    assert np.allclose(p_euclid,utilities.sphere_2_euclid(p_sphere))
