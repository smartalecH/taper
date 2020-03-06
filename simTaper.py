'''
simTaper.py

Simulates mulitple modes for a taper structure and pulls out
a mulitmodal scattering matrix for several modes and frequency points.
'''

import meep as mp
import numpy as np
from matplotlib import pyplot as plt

# Start with geometry
Si = mp.Medium(index=3.4)
SiO2 = mp.Medium(index=1.44)

thickness = 0.22
width1 = 0.5
width2 = 1
length = 4
span = 6
vertices = [
    mp.Vector3(-length/2-span,width1/2),
    mp.Vector3(-length/2,width1/2),
    mp.Vector3(length/2,width2/2),
    mp.Vector3(length/2+span,width2/2),
    mp.Vector3(length/2+span,-width2/2),
    mp.Vector3(length/2,-width2/2),
    mp.Vector3(-length/2,-width1/2),
    mp.Vector3(-length/2-span,-width1/2)
    ]
geometry = [
    mp.Prism(vertices,height=thickness,material=Si) # taper structure
        ]

# Setup domain
resolution = 20
cell_size = mp.Vector3(12,4,0)
boundary_layers = [mp.PML(1.0)]

# Blast it with TE polarized source. Don't worry about an eigenmode source, 
# since we want to measure multiple modes.
sources = [
    mp.EigenModeSource(
    src=mp.GaussianSource(1/1.55,fwidth=0.1/1.55),
    center=[-length/2 - 2],
    size=[0,cell_size.y,cell_size.y],
    eig_parity=mp.ODD_Z+mp.EVEN_Y
    )
]

# Set up simulation object
sim = mp.Simulation(resolution=resolution,
                        cell_size=cell_size,
                        boundary_layers=boundary_layers,
                        geometry=geometry,
                        default_material=SiO2,
                        sources=sources)

# Add the mode monitors
fcen = 1/1.55 # 1.55 microns
df = 0.1*fcen # 10% bandwidth
nf = 30 # number of freqs
m1 = sim.add_mode_monitor(fcen,df,nf,mp.FluxRegion(center=[-length/2,0,0],size=[0,cell_size.y,cell_size.z]))
m2 = sim.add_mode_monitor(fcen,df,nf,mp.FluxRegion(center=[length/2,0,0],size=[0,cell_size.y,cell_size.z]))


# Visualize fields and geometry
sim.run(until=100)
plt.figure()
sim.plot2D(fields=mp.Ez)
plt.title('Ez')
plt.savefig('fields.png')

# Run the rest of the simulation
sim.run(until=600)

# Calculate the scattering params for each waveguide
bands = [1,2,3] # just look at first, second, and third, TE modes
m1_results = sim.get_eigenmode_coefficients(m1,[1],eig_parity=(mp.ODD_Z+mp.EVEN_Y)).alpha
m2_results = sim.get_eigenmode_coefficients(m2,bands,eig_parity=(mp.ODD_Z+mp.EVEN_Y)).alpha

a1 = m1_results[:,:,0] #forward wave
b1 = m1_results[:,:,1] #backward wave
a2 = m2_results[:,:,0] #forward wave
b2 = m2_results[:,:,1] #backward wave

S12_mode1 = a2[0,:] / a1[0,:]
S12_mode2 = a2[1,:] / a1[0,:]
S12_mode3 = a2[2,:] / a1[0,:]

freqs = np.array(mp.get_flux_freqs(m1))

# visualize results
plt.figure()
plt.semilogy(1/freqs,np.abs(S12_mode1)**2,'-o',label='S12 Input Mode 1 Output Mode 1')
plt.semilogy(1/freqs,np.abs(S12_mode2)**2,'-o',label='S12 Input Mode 1 Output Mode 2')
plt.semilogy(1/freqs,np.abs(S12_mode3)**2,'-o',label='S12 Input Mode 1 Output Mode 3')
plt.ylabel('Power')
plt.xlabel('Wavelength (microns)')
plt.legend()
plt.grid(True)
plt.savefig('Results.png')
plt.show()

