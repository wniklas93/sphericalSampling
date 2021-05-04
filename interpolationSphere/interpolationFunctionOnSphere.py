%reload_ext autoreload
%autoreload 2

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from interpolationSphere.sphericalSampling import interpolationOnSphere
from interpolationSphere.sphericalSampling import uniformSampling_unitSphere
from interpolationSphere.sphericalSampling import utilities
################################################################################
#Define Function on Sphere (Directivity Pattern):
N = 2000                    #Number of points
sf = 5                      #sharpening factor for directivity pattern

p_euclid_f, p_sphere_f = uniformSampling_unitSphere.sampleUnitSphere_geometric_fibonacci(N)
phi = p_sphere_f[:,1]
theta = p_sphere_f[:,2]

s = np.sinc(sf*phi/np.pi)*np.sinc(sf*(theta-np.pi/2)/np.pi)

#%%
#Visualize Directivity Patter by Balloon Plot
utilities.SurfacePlot(p_sphere_f[:,1:], s, '')




#%%
