# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 16:37:34 2019

@author: mamartinod
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import glint_simulator_classes as classes

def gaus(x, mu, sigma):
    y = np.exp(-(x - mu)**2/(2 * sigma**2))
    y /= y.sum()
    return y

# =============================================================================
# Fundamental constants
# =============================================================================
c = 3.0E+8 #m/s
h = 6.63E-34 #J.s
flux_ref = 1.8E-8 #Reference flux in W/m^2/micron

# =============================================================================
# Spectral band
# =============================================================================
wl_min, wl_max = 1.45, 1.8 # In µm
bandwidth = wl_max - wl_min

#==============================================================================
# Properties of the detector
#==============================================================================
DIT = 0.02 # in sec
Ndark = 0.015 # in e-/px/img
sigma_ron = 10. # in e-/px/img
nbimg = 10
select_noise = [True, True] # Dark and Readout noises

detector_size = (256, 128) #resp. position axes and wavelength

channel_width = bandwidth / detector_size[1]

QE = 0.9
gainsys = 1./19.12 # ADU/electron
offset = 400

wl_scale = np.arange(wl_min, wl_max, channel_width) #In microns
wave_number = 1/wl_scale # In microns^-1
position_scale = np.arange(0, detector_size[0]) #In pixel

#==============================================================================
# Properties of the pupils and the instrument
#==============================================================================
nb_pupils = 2 # Number of pupils
liste_pupils = np.arange(nb_pupils) # Id of the pupils
diam = 1. #in meter
central_obs = 0.25
surface = np.pi * (diam**2-central_obs**2) / 4 #en m^2
transmission = np.ones((nb_pupils,))*0.01

# =============================================================================
# Properties of the integrated optics and injection
# =============================================================================
kappa_I1_minus = kappa_I1_plus = (1./0.37-1)/2.
kappa_I2_minus = kappa_I2_plus = (1/0.43-1)/2.
splitter_beams = np.array([[0.37, 0.63], [0.43, 0.57], [0.5, 0.5], [0.5, 0.5]]) # one couple per beam. Toward photo and coupler
splitter_beams = np.ones(splitter_beams.shape) * 0.5
rho0 = 0.8

# =============================================================================
# Properties of atmosphere
# =============================================================================
strehl = 1.
rho = 0.8 * strehl
opd0 = 1.55/2.
delta_opd = np.random.normal(0,4., (int(comb(nb_pupils, 2)), nbimg))
opd = opd0 * np.ones(delta_opd.shape) + delta_opd*0
wave_number0 = 1./(2*opd0) # wave number for which the null is minimal

#==============================================================================
# Stars
#==============================================================================
mag = 2.09

na = 0.
visibility = (1 - na) / (1. + na)

# =============================================================================
# Properties of incoming beams on the detector
# =============================================================================
spacing = 1. / (comb(nb_pupils, 2)*2+nb_pupils+1+2)
photo_pos = detector_size[0] * (np.arange(0, nb_pupils)*spacing + spacing)
null_output_pos = photo_pos[-1] + detector_size[0] * spacing + detector_size[0] * (np.arange(0, comb(nb_pupils, 2))*spacing + spacing)
anti_null_output_pos = null_output_pos[-1] + detector_size[0] * spacing + detector_size[0] * (np.arange(0, comb(nb_pupils, 2))*spacing + spacing)
sigma = 1.

# =============================================================================
# Making the clean image
# =============================================================================
image_clean = np.zeros((nbimg, detector_size[0], detector_size[1]))

photometric_channels = np.array([[gaus(position_scale, pos, sigma) for i in range(detector_size[1])] for pos in photo_pos])
null_channels = np.array([[gaus(position_scale, pos, sigma) for i in range(detector_size[1])] for pos in null_output_pos])
anti_null_channels = np.array([[gaus(position_scale, pos, sigma) for i in range(detector_size[1])] for pos in anti_null_output_pos])

photometric_channels = np.transpose(photometric_channels, [0, 2, 1])
null_channels = np.transpose(null_channels, [0, 2, 1])
anti_null_channels = np.transpose(anti_null_channels, [0, 2, 1])

idx_count = 0

for i in liste_pupils:
    N = np.array([QE * surface * channel_width * DIT * np.power(10, -0.4*mag) * flux_ref * wl_scale.mean()*1.E-6 /(h*c)]*detector_size[1])
    photo = np.zeros(image_clean.shape)
    photo[:] = splitter_beams[i][0] * transmission[i] * photometric_channels[i] * N[None, :] * rho
    
    image_clean += photo
    
    for j in liste_pupils[i+1:]:
        beam_i_null = np.zeros(image_clean.shape)
        beam_i_anti_null = np.zeros(image_clean.shape)
        beam_j_null = np.zeros(image_clean.shape)
        beam_j_anti_null = np.zeros(image_clean.shape)
        
        beam_i_null[:] = splitter_beams[i][1]/2. * transmission[i] * null_channels[idx_count] * N[None,:] * rho
        beam_i_anti_null[:] = splitter_beams[i][1]/2. * transmission[i] * anti_null_channels[idx_count] * N[None,:] * rho
        
        beam_j_null[:] = splitter_beams[j][1]/2. * transmission[j] * null_channels[idx_count] * N[None,:] * rho
        beam_j_anti_null[:] = splitter_beams[j][1]/2. * transmission[j] * anti_null_channels[idx_count] * N[None,:] * rho
        
        beams_ij_null = beam_i_null * beam_j_null
        beams_ij_anti_null = beam_i_anti_null * beam_j_anti_null
        
        cosine_null = np.array([np.cos(2*np.pi*(wave_number)*d) for d in opd[idx_count]])
        cosine_anti_null = np.array([np.cos(2*np.pi*(wave_number-wave_number0) * d) for d in opd[idx_count]])
        null_output = beam_i_null + beam_j_null + 2 * np.sqrt(beams_ij_null) * visibility * cosine_null[:,None,:]
        anti_null_output = beam_i_anti_null + beam_j_anti_null + 2 * np.sqrt(beams_ij_anti_null) * visibility * cosine_anti_null[:,None,:]
        
        image_clean += null_output + anti_null_output
        idx_count += 1

data_noisy = classes.Noising(nbimg, image_clean)
data_noisy.addNoise(Ndark, gainsys, offset, sigma_ron, active=select_noise, convert=True)
data = data_noisy.output   

plt.figure()
plt.imshow(image_clean[0], interpolation='none') 
plt.colorbar()
plt.figure()
plt.imshow(data[0], interpolation='none') 
plt.colorbar()

null = image_clean[:,int(round(null_output_pos[0])),:]
antinull = image_clean[:,int(round(anti_null_output_pos[0])),:]

plt.figure();plt.plot(null[0]);plt.grid()
plt.figure();plt.plot(antinull[0]);plt.grid()

null_fft = np.fft.fftshift(np.fft.fft((null), norm='ortho'))
anti_fft = np.fft.fftshift(np.fft.fft((antinull), norm='ortho'))

null_dsp, null_phase = abs(null_fft)**2, np.angle(null_fft)
anti_dsp, anti_phase = abs(anti_fft)**2, np.angle(anti_fft)

plt.figure()
plt.plot(null_phase[0])
plt.grid()

plt.figure()
plt.plot(anti_phase[0])
plt.grid()