# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 16:37:34 2019

@author: mamartinod
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import glint_simulator_classes as classes
from glint_simulator_functions import gaus, setZetaCoeff, save, rv_gen_doubleGauss
from itertools import combinations
import os
import datetime
from timeit import default_timer as time


    
# =============================================================================
# Settings
# =============================================================================
dark_only = False
turbulence = True
save_data = False
nb_block = (0, 10)
nbimg = 6000
select_noise = [True, True] # Dark and Readout noises
conversion = True
path = '/mnt/96980F95980F72D3/glint_data/simulation_opd0/'

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
sigma_ron = 30. # in e-/px/fr

wl_offset = 28 # in pixel, shift of the bandwidth on the detecotr hence a blank in a part of it
detector_size = (344, 96) #resp. position axes and wavelength

channel_width = bandwidth / (detector_size[1])

QE = 0.8
gainsys = 0.5 # ADU/electron
offset = 2750

wl_scale, step = np.linspace(wl_min, wl_max, detector_size[1], endpoint=False, retstep=True) #In microns
wl_scale = wl_scale[::-1]

wave_number = 1/wl_scale # In microns^-1
position_scale = np.arange(0, detector_size[0]) #In pixel

wl_scale_oversampled = np.array([np.linspace(wl_scale[i]-step/2, wl_scale[i]+step/2, 10, endpoint=False) for i in range(wl_scale.size)])
wave_number_oversampled = 1/wl_scale_oversampled

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

rho0 = 0.8

# =============================================================================
# Properties of atmosphere
# =============================================================================

opd0 = 1.602/4.
wave_number0 = 1./(2*opd0) # wave number for which the null is minimal
phase_bias = np.array([-9.801769079200153278e-01, 1.130973355292325344e+00,
                       -9.424777960769379348e-01, 3.129026282975434725e+00,
                       -2.783451091080556772e+00, 1.394867138193868428e+00])#-np.pi/2 # Generic phase bias term

piston_mean, piston_std, jump = 1.602/2, 0.1, 1.
strehl_mean, strehl_std = 0.5, 0.12
#==============================================================================
# Stars
#==============================================================================
mag = 2.
na = 0.0
visibility = (1 - na) / (1. + na)

# =============================================================================
# Properties of incoming beams on the detector
# =============================================================================
channel_positions = np.array([ 33.,  53.,  72.,  92., 112., 132., 151., 171., 191., 211., 230., 250., 270., 290., 309., 329.])
spacing = 1. / (comb(nb_pupils, 2)*2+nb_pupils+1+2)
photo_pos = [channel_positions[15], channel_positions[13], channel_positions[2], channel_positions[0]]

# Order of null:
# 12, 13, 14, 23, 24, 34
# N1, N5, N3, N2, N6, N4
null_output_pos = [channel_positions[11], channel_positions[5], channel_positions[1], channel_positions[3], channel_positions[8], channel_positions[6]]
anti_null_output_pos = [channel_positions[9], channel_positions[7], channel_positions[14], channel_positions[12], channel_positions[10], channel_positions[4]]
sigma = 0.9

# =============================================================================
# Making the clean image
# =============================================================================
photometric_channels = np.array([[gaus(position_scale, pos, sigma) for i in range(detector_size[1])] for pos in photo_pos])
null_channels = np.array([[gaus(position_scale, pos, sigma) for i in range(detector_size[1])] for pos in null_output_pos])
anti_null_channels = np.array([[gaus(position_scale, pos, sigma) for i in range(detector_size[1])] for pos in anti_null_output_pos])

photometric_channels = np.transpose(photometric_channels, [0, 2, 1])
null_channels = np.transpose(null_channels, [0, 2, 1])
anti_null_channels = np.transpose(anti_null_channels, [0, 2, 1])

zeta_minus, zeta_plus = setZetaCoeff(wl_scale, '/mnt/96980F95980F72D3/glint/simulation/zeta_coeff2.hdf5', save=False)

start0 = time()
for bl in range(*nb_block):
    start = time()
    print("Generating block %s / %s"%(bl+1,nb_block[1]))
    if turbulence:
        np.random.seed()
        strehl = np.random.normal(strehl_mean, strehl_std, size=(nb_pupils, nbimg))
        while np.min(strehl) < 0 or np.max(strehl) > 1:
            idx = np.where((strehl<0)|(strehl>1))
            strehl[idx] = np.random.normal(0.5, 0.12, size=idx[0].size)
    else:
        strehl = np.ones((nb_pupils, nbimg))
        
    rho = 0.8 * strehl

    if turbulence:
#        delta_opd = np.random.normal(0, 5, (int(comb(nb_pupils, 2)), nbimg))
        delta_opd = rv_gen_doubleGauss((int(comb(nb_pupils, 2)), nbimg), 0., piston_mean, piston_std, jump)
    else:
        delta_opd = np.zeros((int(comb(nb_pupils, 2)), nbimg))
        
    opd = opd0 * np.ones(delta_opd.shape) + delta_opd
    opd[0] = (0.39999938011169434 - (-1.500000000000056843e-02)) * np.ones(delta_opd.shape[1]) + delta_opd[0] #N1 1st value: optimal piston on seg 29, 2nd value: offset of the center of coherent envelop
    opd[3] = (0 - 9.653999999999999915e+00) * np.ones(delta_opd.shape[1]) + delta_opd[3] #N2
    opd[5] = (-1.299999475479126 - (-2.721000000000000085e+00)) * np.ones(delta_opd.shape[1]) + delta_opd[5] #N4
#    opd_ramp = np.linspace(-50,+50,nbimg, endpoint=False)
#    opd[0,:] += opd_ramp
    
    # Order of pairs of beams: 12, 13, 14, 23, 24, 34
    opd[1] = opd[0] + opd[3] # 13 = 12 + 23 i.e. N5
    opd[2] = opd[1] + opd[5] # 14 = 13 + 34 i.e. N3
    opd[4] = opd[3] + opd[5] # 24 = 23 + 34 i.e. N6
    
    image_clean = np.zeros((nbimg, detector_size[0], detector_size[1]))
    
    if not dark_only:
        idx_count = 0
        for i in liste_pupils:
            N = QE * surface * channel_width * DIT * np.power(10, -0.4*mag) * flux_ref * wl_scale.mean()*1.E-6 /(h*c) * np.ones(detector_size[1])
            photo = np.zeros(image_clean.shape)
            photo[:] = transmission[i] * photometric_channels[i] * N[None, :] * rho[i,:,None,None]
            
            image_clean += photo
            
            for j in liste_pupils[i+1:]:
                beam_i_null = np.zeros(image_clean.shape)
                beam_i_anti_null = np.zeros(image_clean.shape)
                beam_j_null = np.zeros(image_clean.shape)
                beam_j_anti_null = np.zeros(image_clean.shape)
                
                zeta_minus_i = zeta_minus[idx_count,0]
                zeta_minus_j = zeta_minus[idx_count,1]
                zeta_plus_i = zeta_plus[idx_count,0]
                zeta_plus_j = zeta_plus[idx_count,1]
                
                beam_i_null[:] = transmission[i] * null_channels[idx_count] * N[None,:] * rho[i,:,None,None]
                beam_i_anti_null[:] = transmission[i] * anti_null_channels[idx_count] * N[None,:] * rho[i,:,None,None]
                
                beam_j_null[:] = transmission[j] * null_channels[idx_count] * N[None,:] * rho[j,:,None,None]
                beam_j_anti_null[:] = transmission[j] * anti_null_channels[idx_count] * N[None,:] * rho[j,:,None,None]
                
                beams_ij_null = beam_i_null * beam_j_null
                beams_ij_anti_null = beam_i_anti_null * beam_j_anti_null
                

#                sine = np.array([np.sin(2*np.pi*wave_number*d + phase_bias) for d in opd[idx_count]])
                sine = np.array([np.sin(2*np.pi*wave_number_oversampled*d + phase_bias[idx_count]) for d in opd[idx_count]])
                sine = np.mean(sine, axis=2)
                null_output = beam_i_null * zeta_minus_i + beam_j_null * zeta_minus_j -\
                2*np.sqrt(beams_ij_null) * np.sqrt(zeta_minus_i * zeta_minus_j) * visibility * sine[:,None,:]
                
                anti_null_output = beam_i_anti_null * zeta_plus_i + beam_j_anti_null * zeta_plus_j + \
                2*np.sqrt(beams_ij_anti_null) * np.sqrt(zeta_plus_i * zeta_plus_j) * visibility * sine[:,None,:]

                image_clean += null_output + anti_null_output
                idx_count += 1
        
        image_clean[:,:,:wl_offset] = 0
    
    # =============================================================================
    # Noising frames
    # =============================================================================
    data_noisy = classes.Noising(nbimg, image_clean)
    data_noisy.addNoise(Ndark, gainsys, offset, sigma_ron, active=select_noise, convert=conversion)
    data = data_noisy.output
    
    # =============================================================================
    # Save data
    # =============================================================================
    if save_data:
        if not os.path.exists(path):
            os.makedirs(path)
            
        if not dark_only:
            path2 = path + 'simu_'+str(mag)+'_%04d.mat'%(bl+1)
        else:
            path2 = path + 'dark_%04d.mat'%(bl+1)
        
        date = datetime.datetime.utcnow().isoformat()
        save(np.transpose(data, axes=(0,2,1)), path2, date, DIT, na, mag, nbimg, [piston_mean, piston_std, jump], [strehl_mean, strehl_std])
    
    stop = time()
    print('Last: %.3f'%(stop-start))

stop0 = time()
print('Total last: %.3f'%(stop0-start0))

# =============================================================================
# Miscelleanous
# =============================================================================
output = data
null = output[:,int(round(null_output_pos[0])),:]
antinull = output[:,int(round(anti_null_output_pos[0])),:]
p = np.array([output[:,int(round(photo_pos_i)),:] for photo_pos_i in photo_pos])

for i in range(min(5,nbimg)):
    plt.figure()
    plt.subplot(131)
    plt.imshow(data[i], interpolation='none', aspect='auto', extent=[wl_scale[0], wl_scale[-1], detector_size[0], 0],) 
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(image_clean[i], interpolation='none', aspect='auto', extent=[wl_scale[0], wl_scale[-1], detector_size[0], 0]) 
    plt.colorbar()
    plt.subplot(133)
    plt.plot(wl_scale, p[0,i], label='P1')
    plt.plot(wl_scale, p[1,i], 'o', label='P2')
    plt.plot(wl_scale, p[2,i], '+', label='P3')
    plt.plot(wl_scale, p[3,i], 'x', label='P4')
    plt.plot(wl_scale, null[i], label='null')
    plt.plot(wl_scale, antinull[i], '.', label='antinull')
    plt.grid()
    plt.legend(loc='best')

plt.figure()
plt.plot(opd[0,:], null[:,55], label='null')
plt.grid()
plt.legend(loc='best')

plt.figure()
plt.plot(opd[0,:], null[:,20:60].sum(axis=1), '-', label='sum of null')
plt.grid()
plt.xlim(-50,50)
plt.legend(loc='best')

from matplotlib import animation

fig = plt.figure()
ax = plt.axes()
time_text = ax.text(0.05, 0.01, '', transform=ax.transAxes, color='w')

im = plt.imshow(output[0],interpolation='none')
# initialization function: plot the background of each frame
def init():
    im.set_data(output[0])
    time_text.set_text('')
    return [im] + [time_text]

# animation function.  This is called sequentially
def animate(i):
    im.set_array(output[i])
    time_text.set_text('Frame %s/%s'%(i+1, image_clean.shape[0]))
    return [im] + [time_text]

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=output.shape[0], interval=200, blit=True)