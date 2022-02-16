# %%
%reload_ext autoreload
%autoreload 2

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from sphericalSampling import resamplingOnSphere
from sphericalSampling import uniformSampling_unitSphere
from sphericalSampling import utilities
# %%
################################################################################
#Define Function on Sphere (Directivity Pattern):
soundPressureFunc = lambda phi, theta, sf: np.sinc(sf*phi/np.pi)*np.sinc(sf*(theta-np.pi/2)/np.pi)

N = 2000                     #Number of points
sf = 5                       #sharpening factor for directivity pattern

p_euclid_f, p_sphere_f = uniformSampling_unitSphere.sampleUnitSphere_geometric_fibonacci(N)
phi = p_sphere_f[:,1]
theta = p_sphere_f[:,2]

s = soundPressureFunc(phi, theta, sf)

#Visualize Directivity Patter by Balloon Plot
utilities.SurfacePlot(p_sphere_f[:,1:], s, 'Sampled Sound Pressure')

#%%
#Define interpolation/resampling points:
p_euclid_i, p_sphere_i = uniformSampling_unitSphere.sampleUnitSphere_statistical_normal(N)
s_ref = soundPressureFunc(p_sphere_i[:,1], p_sphere_i[:,2], sf)
utilities.SurfacePlot(p_sphere_i[:,1:], s_ref, 'Sampled Sound Pressure (Interpolation Reference)')

#%%
#Interpolation/Resampling:
#Resampling by using spherical harmonics
si_sh = resamplingOnSphere.interpolation_sphericalHarmonics(p_sphere_i[:,1:],
                                                               p_sphere_f[:,1:],
                                                               s)
#Resampling by using spherical splines
k_f_polyh_spline = resamplingOnSphere.kernel('TPF', s=3)
si_l = resamplingOnSphere.interpolation_lagrange(p_euclid_i,
                                                    p_euclid_f,
                                                    s,
                                                    k_f_polyh_spline)

utilities.SurfacePlot(p_sphere_i[:,1:], si_l, 'Sampled Sound Pressure (Interpolant)')


#%%
#Error Evaluation:
err_sh = resamplingOnSphere.max_squaredError(s_ref, si_sh)
err_l = resamplingOnSphere.max_squaredError(s_ref, si_l)
err_l
#%%

