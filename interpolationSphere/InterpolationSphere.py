%reload_ext autoreload
%autoreload 2

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from interpolationSphere.sphericalSampling import uniformSampling_unitSphere
from interpolationSphere.sphericalSampling import utilities
################################################################################
#Uniformly Sampling using Normal Distribution Method:
N = 1000
p_euclide, p_sphere = uniformSampling_unitSphere.sampleUnitSphere_statistical_normal(N)
utilities.plot_3D(utilities.sphere_2_euclid(p_sphere), "Uniformly Sampling (Normal Distribution Method)")
#%%
#Uniformly Sampling of Unit Disc using Fibonacci Lattice:
N = 500
p_euclide_2d, p_pol = uniformSampling_unitSphere.fibonacci_lattice_unitDisc(N)
utilities.plot_2D(p_euclide_2d, "Uniformly Sampling of unit Disc (Fibonacci Lattice)")


#%%
#Uniformly Sampling of Unit Square using Fibonacci Lattice:
N = 100
p_euclide_2d = uniformSampling_unitSphere.fibonacci_lattice_unitSquare(N)
utilities.plot_2D(p_euclide_2d, "Uniformly Sampling of Unit Square (Fibonacci Lattice)")
#%%
N = 2000
p_euclide, p_sphere = uniformSampling_unitSphere.sampleUnitSphere_geometric_fibonacci(N)
utilities.plot_3D(utilities.sphere_2_euclid(p_sphere), "Uniformly Sampling (Fibonacci Lattice)")










#%%
