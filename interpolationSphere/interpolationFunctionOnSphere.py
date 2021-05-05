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
soundPressureFunc = lambda phi, theta, sf: np.sinc(sf*phi/np.pi)*np.sinc(sf*(theta-np.pi/2)/np.pi)

N = 2000                     #Number of points
sf = 5                      #sharpening factor for directivity pattern

p_euclid_f, p_sphere_f = uniformSampling_unitSphere.sampleUnitSphere_geometric_fibonacci(N)
phi = p_sphere_f[:,1]
theta = p_sphere_f[:,2]

s = soundPressureFunc(phi, theta, sf)

#Visualize Directivity Patter by Balloon Plot
#utilities.SurfacePlot(p_sphere_f[:,1:], s, 'Sampled Sound Pressure')

#%%
#Define Interpolation points:
p_euclid_i, p_sphere_i = uniformSampling_unitSphere.sampleUnitSphere_statistical_normal(N)
s_ref = soundPressureFunc(phi, theta, sf)
#utilities.SurfacePlot(p_sphere_f[:,1:], s, 'Sampled Sound Pressure (Interpolation Reference)')

#%%
#Interpolation:
si = interpolationOnSphere.interpolation_sphericalHarmonics(p_sphere_i[1:,:],
                                                            p_sphere_f[1:,:],
                                                            s)





#%%
