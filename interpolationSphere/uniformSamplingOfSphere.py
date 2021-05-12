%reload_ext autoreload
%autoreload 2

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from interpolationSphere.sphericalSampling import uniformSampling_unitSphere
from interpolationSphere.sphericalSampling import utilities
################################################################################
#Uniformly Sampling using Normal Distribution Method:
N = 2000
p_euclid_n, p_sphere_n = uniformSampling_unitSphere.sampleUnitSphere_statistical_normal(N)
utilities.plot_3D(utilities.sphere_2_euclid(p_sphere_n), "Uniformly Sampling (Normal Distribution Method)")
#%%
#Uniformly Sampling of Unit Disc using Fibonacci Lattice:
N = 500
p_euclid_2d, p_pol = uniformSampling_unitSphere.fibonacci_lattice_unitDisc(N)
#utilities.plot_2D(p_euclid_2d, "Uniformly Sampling of unit Disc (Fibonacci Lattice)")


#%%
#Uniformly Sampling of Unit Square using Fibonacci Lattice:
N = 100
p_euclid_2d = uniformSampling_unitSphere.fibonacci_lattice_unitSquare(N)
#utilities.plot_2D(p_euclid_2d, "Uniformly Sampling of Unit Square (Fibonacci Lattice)")
#%%
#Uniformly Sampling of Unit Sphere using Fibonacci Lattice:
N = 2000
p_euclid_f, p_sphere_f = uniformSampling_unitSphere.sampleUnitSphere_geometric_fibonacci(N)
utilities.plot_3D(utilities.sphere_2_euclid(p_sphere_f), "Uniformly Sampling (Fibonacci Lattice)")

#%%
#Uniformly Sampling of Unit Sphere using Wolfgang's method:
N = 2000
p_euclid_W, p_sphere_W = uniformSampling_unitSphere.sampleUnitSphere_geometric_Wolfgang(N)
utilities.plot_3D(utilities.sphere_2_euclid(p_sphere_W), "Uniformly Sampling (Wolfgang)")
#%%
#Evaluate Uniformity of distribution over S^2 using Spherical Cap Discrepancy
N = 40
Np = np.logspace(1,4, N, endpoint=True, dtype=int)
d_fibonacci = np.zeros((N,))
d_normal = np.zeros((N,))
d_wolfgang = np.zeros((N,))
d_aistleitner = Np**(-3/4)

for i,n in enumerate(Np):
    p_euclid_f, _ = uniformSampling_unitSphere.sampleUnitSphere_geometric_fibonacci(n)
    p_euclid_n, _ = uniformSampling_unitSphere.sampleUnitSphere_statistical_normal(n)
    p_euclid_W, _ = uniformSampling_unitSphere.sampleUnitSphere_geometric_Wolfgang(n)

    d_fibonacci[i] = uniformSampling_unitSphere.spherical_cap_discrepancy(p_euclid_f)
    d_normal[i] = uniformSampling_unitSphere.spherical_cap_discrepancy(p_euclid_n)
    d_wolfgang[i] = uniformSampling_unitSphere.spherical_cap_discrepancy(p_euclid_W)


fig = plt.figure(figsize=(10,8))
plt.title('Spherical Cap Discrepancy')
plt.grid(True)
plt.scatter(Np, d_fibonacci, label='Spherical Fibonacci Lattice')
plt.scatter(Np, d_normal, label='Spherical Normal-Distribution Lattice')
plt.scatter(Np, d_wolfgang, label='Spherical Lattice (Wolfgang)')
plt.plot(Np, d_aistleitner, label='Aistleitner et al.')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Points on Sphere')
plt.ylabel('Spherical Cap Discrepancy')
plt.legend()


#%%

p_sphere_f[:,1][:10]
