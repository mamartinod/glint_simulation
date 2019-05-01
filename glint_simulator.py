# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 16:37:34 2019

@author: mamartinod
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import glint_simulator_classes as classes
from itertools import combinations
import h5py
import os
import datetime

def gaus(x, mu, sigma):
    y = np.exp(-(x - mu)**2/(2 * sigma**2))
    y /= y.sum()
    return y

def chromatic_splitters(ratios, wl_scale, wl0, slope):
    chroma_ratios = np.zeros((ratios.shape[0], 2, wl_scale.size))
    chromatism = slope * (wl_scale - wl0)
    chroma_ratios[:,0] = ratios[:,0,None] + chromatism[None,:]
    chroma_ratios[:,1] = 1 - chroma_ratios[:,0]
        
    return chroma_ratios

def save(arr, path, date, DIT, na, mag):
    # Check if saved file exist
    if os.path.exists(path):
        opening_mode = 'w' # Overwright the whole existing file.
    else:
        opening_mode = 'a' # Create a new file at "path"
        
    with h5py.File(path, opening_mode) as f:
        f.attrs['origin'] = 'sim'
        f.attrs['date'] = date
        f.attrs['nbimg'] = nbimg 
        f.attrs['DIT'] = DIT
        f.attrs['Na'] = na
        f.attrs['mag'] = mag
        f.create_dataset('imagedata', data=arr)

# =============================================================================
# Execution mode        
# =============================================================================
dark_only = False
save_data = True
nb_block = 5
# =============================================================================
# Fundamental constants
# =============================================================================
c = 3.0E+8 #m/s
h = 6.63E-34 #J.s
flux_ref = 1.8E-8 #Reference flux in W/m^2/micron

# =============================================================================
# Spectral band
# =============================================================================
wl_min, wl_max = 1.352, 1.832 # In µm
bandwidth = wl_max - wl_min

#==============================================================================
# Properties of the detector
#==============================================================================
DIT = 0.002 # in sec
Ndark = 600 # in e-/px/s
Ndark *= DIT # in e-/px/fr
sigma_ron = 10. # in e-/px/fr
nbimg = 1000
select_noise = [True, True] # Dark and Readout noises

detector_size = (344, 96) #resp. position axes and wavelength

channel_width = bandwidth / detector_size[1]

QE = 0.8
gainsys = 1./3. # ADU/electron
offset = 2750

wl_scale = np.arange(wl_min, wl_max, channel_width)[::-1] #In microns
wave_number = 1/wl_scale # In microns^-1
position_scale = np.arange(0, detector_size[0]) #In pixel

#==============================================================================
# Properties of the pupils and the instrument
#==============================================================================
nb_pupils = 4 # Number of pupils
liste_pupils = np.arange(nb_pupils) # Id of the pupils
diam = 1. #in meter
central_obs = 0.25
surface = np.pi * (diam**2-central_obs**2) / 4 #en m^2
transmission = np.ones((nb_pupils,))*0.01

# =============================================================================
# Properties of the integrated optics and injection
# =============================================================================
beam_comb = [str(i)+''+str(j) for i,j in combinations(np.arange(4), 2)]
beam_splitter = np.array([[0.37, 0.63], [0.43, 0.57], [0.5, 0.5], [0.5, 0.5]]) # one couple per beam. Toward photo and coupler
beam_splitter = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]) # one couple per beam. Toward photo and coupler
beam_splitter = chromatic_splitters(beam_splitter, wl_scale, 1.55, 0.)

# Setting the phase of the coupler
couplers_splitter = np.ones((6,2)) * np.pi/4 # Toward null and anti-null
couplers_splitter = chromatic_splitters(couplers_splitter, wl_scale, 1.55, 0.)

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
phase_bias = -np.pi/2 # Generic phase bias term

#==============================================================================
# Stars
#==============================================================================
mag = 2.1
na = 0.
visibility = (1 - na) / (1. + na)

# =============================================================================
# Properties of incoming beams on the detector
# =============================================================================
channel_positions = np.array([ 33.,  53.,  72.,  92., 112., 132., 151., 171., 191., 211., 230., 250., 270., 290., 309., 329.])
spacing = 1. / (comb(nb_pupils, 2)*2+nb_pupils+1+2)
photo_pos = [channel_positions[15], channel_positions[13], channel_positions[2], channel_positions[0]]
null_output_pos = [channel_positions[11], channel_positions[3], channel_positions[1], channel_positions[6], channel_positions[5], channel_positions[8]]
anti_null_output_pos = [channel_positions[9], channel_positions[12], channel_positions[14], channel_positions[4], channel_positions[7], channel_positions[10]]
sigma = 1.3

# =============================================================================
# Making the clean image
# =============================================================================
photometric_channels = np.array([[gaus(position_scale, pos, sigma) for i in range(detector_size[1])] for pos in photo_pos])
null_channels = np.array([[gaus(position_scale, pos, sigma) for i in range(detector_size[1])] for pos in null_output_pos])
anti_null_channels = np.array([[gaus(position_scale, pos, sigma) for i in range(detector_size[1])] for pos in anti_null_output_pos])

photometric_channels = np.transpose(photometric_channels, [0, 2, 1])
null_channels = np.transpose(null_channels, [0, 2, 1])
anti_null_channels = np.transpose(anti_null_channels, [0, 2, 1])

for bl in range(nb_block):
    print("Generating block %s / %s"%(bl+1,nb_block))
    image_clean = np.zeros((nbimg, detector_size[0], detector_size[1]))
    
    if not dark_only:
        idx_count = 0
        for i in liste_pupils:
            N = np.array([QE * surface * channel_width * DIT * np.power(10, -0.4*mag) * flux_ref * wl_scale.mean()*1.E-6 /(h*c)]*detector_size[1])
            photo = np.zeros(image_clean.shape)
            photo[:] = beam_splitter[i][0] * transmission[i] * photometric_channels[i] * N[None, :] * rho
            
            image_clean += photo
            
            for j in liste_pupils[i+1:]:
                kappa_l = couplers_splitter[idx_count,0,:] # Asymmetric coupler is not implemented yet
                
                beam_i_null = np.zeros(image_clean.shape)
                beam_i_anti_null = np.zeros(image_clean.shape)
                beam_j_null = np.zeros(image_clean.shape)
                beam_j_anti_null = np.zeros(image_clean.shape)
                
                beam_i_null[:] = beam_splitter[i][1] * transmission[i] * null_channels[idx_count] * N[None,:] * rho
                beam_i_anti_null[:] = beam_splitter[i][1] * transmission[i] * anti_null_channels[idx_count] * N[None,:] * rho
                
                beam_j_null[:] = beam_splitter[j][1] * transmission[j] * null_channels[idx_count] * N[None,:] * rho
                beam_j_anti_null[:] = beam_splitter[j][1] * transmission[j] * anti_null_channels[idx_count] * N[None,:] * rho
                
                beams_ij_null = beam_i_null * beam_j_null
                beams_ij_anti_null = beam_i_anti_null * beam_j_anti_null
                
                sine = np.array([np.sin(2*np.pi*(wave_number)*d + phase_bias) for d in opd[idx_count]])
                null_output = beam_i_null * np.sin(kappa_l)**2 + beam_j_null * np.cos(kappa_l)**2 -\
                np.sqrt(beams_ij_null) * np.sin(2 * kappa_l) * visibility * sine[:,None,:]
                
                anti_null_output = beam_i_anti_null * np.cos(kappa_l)**2 + beam_j_anti_null * np.sin(kappa_l)**2 + \
                np.sqrt(beams_ij_anti_null) * np.sin(2 * kappa_l) * visibility * sine[:,None,:]
                
                image_clean += null_output + anti_null_output
                idx_count += 1
    
    # =============================================================================
    # Noising frames
    # =============================================================================
    data_noisy = classes.Noising(nbimg, image_clean)
    data_noisy.addNoise(Ndark, gainsys, offset, sigma_ron, active=select_noise, convert=True)
    data = data_noisy.output   
    
    # =============================================================================
    # Save data
    # =============================================================================
    if save_data:
        if not dark_only:
            path = '/mnt/96980F95980F72D3/glint_data/simulation/simu_'+str(mag)+'_%04d.mat'%(bl+1)
        else:
            path = '/mnt/96980F95980F72D3/glint_data/simulation/dark.mat'
        
        date = datetime.datetime.utcnow().isoformat()
        save(np.transpose(data, axes=(0,2,1)), path, date, DIT, na, mag)

# =============================================================================
# Miscelleanous
# =============================================================================
plt.figure()
plt.imshow(image_clean[0], interpolation='none', aspect='auto', extent=[wl_scale[0], wl_scale[-1], detector_size[0], 0]) 
plt.colorbar()
plt.figure()
plt.imshow(data[0], interpolation='none', aspect='auto', extent=[wl_scale[0], wl_scale[-1], detector_size[0], 0]) 
plt.colorbar()

null = image_clean[:,int(round(null_output_pos[0])),:]
antinull = image_clean[:,int(round(anti_null_output_pos[0])),:]
p = np.array([image_clean[:,int(round(photo_pos_i)),:] for photo_pos_i in photo_pos])

plt.figure()
plt.plot(wl_scale, p[0,0], label='P1')
plt.plot(wl_scale, p[1,0], 'o', label='P2')
plt.plot(wl_scale, null[0], label='null')
plt.plot(wl_scale, antinull[0], '.', label='antinull')
plt.grid()
plt.legend(loc='best')

null_fft = np.fft.fftshift(np.fft.fft((null), norm='ortho'))
anti_fft = np.fft.fftshift(np.fft.fft((antinull), norm='ortho'))

null_dsp, null_phase = abs(null_fft)**2, np.angle(null_fft)
anti_dsp, anti_phase = abs(anti_fft)**2, np.angle(anti_fft)

#plt.figure()
#plt.plot(null_phase[0])
#plt.grid()
#
#plt.figure()
#plt.plot(anti_phase[0])
#plt.grid()