# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 16:37:34 2019

@author: mamartinod
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import glint_simulator_classes as classes
from glint_simulator_functions import gaus, setZetaCoeff, save, rv_gen_doubleGauss, save_segment_positions
from itertools import combinations
import os
import datetime
from timeit import default_timer as time

# =============================================================================
# Settings
# =============================================================================
activate_dark_only = False
activate_scan_null = False
activate_turbulence = False
activate_phase_bias = True
activate_opd_bias = True
activate_zeta = True
activate_oversampling = True
activate_crosstalk = False
select_noise = [False, False] # Dark and Readout noises
conversion = False
nb_block = (0, 1)
nbimg = 1
save_data = False
path = '/mnt/96980F95980F72D3/glint_data/20191015_simulation/'
save_segment_pos_path = '/mnt/96980F95980F72D3/glint/reduction/calibration_params_simu/N1N4_opti.mat'
auto_talk_beams = [0, 1]
scanned_segment = 0
scan_bounds = (-2500, 2500) # In nm

# =============================================================================
# Fundamental constants
# =============================================================================
c = 3.0E+8 #m/s
h = 6.63E-34 #J.s
flux_ref = 1.8E-11 #Reference flux in W/m^2/nm
wl0 = 1524.5#1602
opd0 = wl0/4.

# =============================================================================
# Spectral band
# =============================================================================
wl_min, wl_max = 1357, 1832 # In nm
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

step = 5 #In nm
wl_scale = np.arange(wl_min, wl_max+5,5, dtype=np.float64) #In nm
wl_scale = wl_scale[::-1]

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

rho0 = 0.8

instrumental_offsets = np.loadtxt('/mnt/96980F95980F72D3/glint/reduction/calibration_params_simu/4WG_opd0_and_phase_simu.txt')
 # Order of pairs of beams: 12, 31, 14, 23, 42, 34 (N1, N5, N3, N2, N6, N4)
if activate_opd_bias:
    opd_bias = instrumental_offsets[:,0].copy()
    opd_bias[0] = instrumental_offsets[0,0] # N1
    opd_bias[1] = instrumental_offsets[4,0] # N5
    opd_bias[2] = instrumental_offsets[2,0] # N3
    opd_bias[3] = instrumental_offsets[1,0] # N2
    opd_bias[4] = instrumental_offsets[5,0] # N6
    opd_bias[5] = instrumental_offsets[3,0] # N4
else:
    print('no opd bias')
    opd_bias = -np.ones(6,) * opd0
    opd_bias[1] = -opd_bias[3] - opd_bias[0]  # 31 = 32 + 21 = -23 - 12 i.e. N5
    opd_bias[2] = -opd_bias[1] + opd_bias[5] # 14 = 13 + 34 = -31 + 34 i.e. N3
    opd_bias[4] = -opd_bias[5] - opd_bias[3] # 42 = 43 + 32 = -34 - 23 i.e. N6
    
if activate_phase_bias:
    phase_bias = instrumental_offsets[:,1].copy()
    phase_bias[0] = instrumental_offsets[0,1]
    phase_bias[1] = instrumental_offsets[4,1]
    phase_bias[2] = instrumental_offsets[2,1]
    phase_bias[3] = instrumental_offsets[1,1]
    phase_bias[4] = instrumental_offsets[5,1]
    phase_bias[5] = instrumental_offsets[3,1]
else:
    phase_bias = np.zeros(6,)
    
segment_positions = np.zeros(4,)
segment_positions[0] = 600
segment_positions[3] = 2680

# =============================================================================
# Properties of atmosphere
# =============================================================================
piston_mu = 20
piston_shift, jump = wl0/2, 0.
piston_std = 40.
strehl_mean, strehl_std = 0.5, 0.12

# =============================================================================
# Crosstalk
# =============================================================================
if activate_crosstalk:
    cross_talk_factor = 0.01
    crosstalk_opd = np.ones(4,) * 60000. + np.array([0, -1000, 1500, 25000])
    crosstalk_phase = np.ones(4,) * np.pi * (1-2/1550 * crosstalk_opd[0])
    crosstalk_phase = crosstalk_phase #+ np.array([0, -np.pi, np.pi/4, np.pi])
    arg = 2*np.pi*wave_number[None,:]*crosstalk_opd[:,None] + crosstalk_phase[:,None]
    wiggles = np.cos(arg)
    crosstalk = 1 + cross_talk_factor + 2*cross_talk_factor**0.5 * wiggles
#    normalization = (arg[:,-1][:,None]-arg[:,0][:,None])/np.mean(np.diff(arg, axis=1), axis=1)[:,None]
#    crosstalk /= normalization

#==============================================================================
# Stars
#==============================================================================
mag = 0.
na = np.array([0.2, 1., 1., 1., 1., 0.])
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
null_labels = ['N1', 'N5', 'N3', 'N2', 'N6', 'N4']
null_output_pos = [channel_positions[11], channel_positions[5], channel_positions[1], channel_positions[3], channel_positions[8], channel_positions[6]]
anti_null_output_pos = [channel_positions[9], channel_positions[7], channel_positions[14], channel_positions[12], channel_positions[10], channel_positions[4]]
sigma = 0.9
null_photo_map = np.array([[0,1], [2,0], [0,3], [1,2], [3,1], [2,3]])

# =============================================================================
# Making the clean image
# =============================================================================
photometric_channels = np.array([[gaus(position_scale, pos, sigma) for i in range(detector_size[1])] for pos in photo_pos])
null_channels = np.array([[gaus(position_scale, pos, sigma) for i in range(detector_size[1])] for pos in null_output_pos])
anti_null_channels = np.array([[gaus(position_scale, pos, sigma) for i in range(detector_size[1])] for pos in anti_null_output_pos])

photometric_channels = np.transpose(photometric_channels, [0, 2, 1])
null_channels = np.transpose(null_channels, [0, 2, 1])
anti_null_channels = np.transpose(anti_null_channels, [0, 2, 1])

#photometric_channels /= np.sum(photometric_channels, axis=(-1,-2))[:,None,None]
#null_channels /= np.sum(null_channels, axis=(-1,-2))[:,None,None]
#anti_null_channels /= np.sum(anti_null_channels, axis=(-1,-2))[:,None,None]

if activate_zeta:
    zeta_minus, zeta_plus = setZetaCoeff(wl_scale, '/mnt/96980F95980F72D3/glint/reduction/calibration_params_simu/zeta_coeff_simu.hdf5', save=False)
else:
    zeta_minus = np.ones((6,2,96))
    zeta_plus = np.ones((6,2,96))
    

if activate_scan_null:
    scan_range = np.linspace(scan_bounds[0], scan_bounds[1], 1001)
    nbimg = scan_range.size
    activate_turbulence = False
    activate_dark_only = False
    select_noise = [False, False] # Dark and Readout noises
    conversion = False
    nb_block = (0, 1)
    save_data = False
    
start0 = time()
for bl in range(*nb_block):
    start = time()
    print("Generating block %s / %s"%(bl+1,nb_block[1]))
    if activate_scan_null:
        segment_positions = np.zeros((4, scan_range.size))
        segment_positions[scanned_segment] = scan_range
    
    if activate_turbulence:
        strehl = np.random.normal(strehl_mean, strehl_std, size=(nb_pupils, nbimg))
        while np.min(strehl) < 0 or np.max(strehl) > 1:
            idx = np.where((strehl<0)|(strehl>1))
            strehl[idx] = np.random.normal(0.5, 0.12, size=idx[0].size)
    else:
        strehl = np.ones((nb_pupils, nbimg))
        
#    strehl[:] = strehl[-1]
    rho = rho0 * strehl

    if activate_turbulence:
        if jump != 0:
            piston_pupils = rv_gen_doubleGauss((4, nbimg), 0., piston_shift, piston_std, jump)
        else:
            piston_pupils = np.random.normal(piston_mu, piston_std, (4, nbimg))
    else:
        piston_pupils = np.zeros((4, nbimg))
        
    opd = np.ones((int(comb(nb_pupils, 2)), nbimg))
#    opd_ramp = np.linspace(-50000,+50000,nbimg, endpoint=False)
#    opd[0,:] += opd_ramp
    
    # Order of pairs of beams: 12, 31, 14, 23, 42, 34 (N1, N5, N3, N2, N6, N4)
    # Independant pairs: N1: 12, N2: 23, N4: 34
    opd[0] = -opd_bias[0] + piston_pupils[0] - piston_pupils[1] + segment_positions[0] - segment_positions[1] # N1: 12
    opd[3] = -opd_bias[3] + piston_pupils[1] - piston_pupils[2] + segment_positions[1] - segment_positions[2] # N2: 23
    opd[5] = -opd_bias[5] + piston_pupils[2] - piston_pupils[3] + segment_positions[2] - segment_positions[3] # N4: 34
    
#    opd[1] = -opd[3] - opd[0]  # 31 = 32 + 21 = -23 - 12 i.e. N5
#    opd[2] = -opd[1] + opd[5] # 14 = 13 + 34 = -31 + 34 i.e. N3
#    opd[4] = -opd[5] - opd[3] # 42 = 43 + 32 = -34 - 23 i.e. N6
    
    opd[1] = -opd_bias[1] + piston_pupils[2] - piston_pupils[0] + segment_positions[2] - segment_positions[0]# 31 = 32 + 21 = -23 - 12 i.e. N5
    opd[2] = -opd_bias[2] + piston_pupils[0] - piston_pupils[3] + segment_positions[0] - segment_positions[3]# 14 = 13 + 34 = -31 + 34 i.e. N3
    opd[4] = -opd_bias[4] + piston_pupils[3] - piston_pupils[1] + segment_positions[3] - segment_positions[1]# 42 = 43 + 32 = -34 - 23 i.e. N6
#    opd[:] = opd[-1]
    
    image_clean = np.zeros((nbimg, detector_size[0], detector_size[1]))
    
    if not activate_dark_only:
        idx_count = 0
        for i in liste_pupils:
            N = QE * surface * channel_width * DIT * np.power(10, -0.4*mag) * flux_ref * wl_scale.mean()*1.E-9 /(h*c) * np.ones(detector_size[1])
            photo = np.zeros(image_clean.shape)
            photo[:] = transmission[i] * photometric_channels[i] * N[None, :] * rho[i,:,None,None]
            
            if activate_crosstalk:
                if i in auto_talk_beams:
                    ct = crosstalk[i]
                    photo = photo * ct[None,None,:]
            
            image_clean += photo
            
            for j in liste_pupils[i+1:]:
                beam_i_null = np.zeros(image_clean.shape)
                beam_i_anti_null = np.zeros(image_clean.shape)
                beam_j_null = np.zeros(image_clean.shape)
                beam_j_anti_null = np.zeros(image_clean.shape)
                
                zeta_null_i = zeta_minus[idx_count,0]
                zeta_null_j = zeta_minus[idx_count,1]
                zeta_antinull_i = zeta_plus[idx_count,0]
                zeta_antinull_j = zeta_plus[idx_count,1]
                
                beam_i_null[:] = transmission[i] * null_channels[idx_count] * N[None,:] * rho[i,:,None,None]
                beam_i_anti_null[:] = transmission[i] * anti_null_channels[idx_count] * N[None,:] * rho[i,:,None,None]
                
                beam_j_null[:] = transmission[j] * null_channels[idx_count] * N[None,:] * rho[j,:,None,None]
                beam_j_anti_null[:] = transmission[j] * anti_null_channels[idx_count] * N[None,:] * rho[j,:,None,None]
                
                if activate_crosstalk:
                    if i in auto_talk_beams:
                        ct = crosstalk[i]
                        beam_i_null = beam_i_null * ct[None,None,:]
                        beam_i_anti_null = beam_i_anti_null * ct[None,None,:]
                    if j in auto_talk_beams:
                        ct = crosstalk[j]
                        beam_j_null = beam_j_null * ct[None,None,:]
                        beam_j_anti_null = beam_j_anti_null * ct[None,None,:]
                
                beams_ij_null = beam_i_null * beam_j_null
                beams_ij_anti_null = beam_i_anti_null * beam_j_anti_null
                
                sine = np.array([np.sin(2*np.pi*wave_number*d + phase_bias[idx_count]) for d in opd[idx_count]])
                if activate_oversampling:
                    dwn = abs(1/(wl_scale+step/2) - 1/(wl_scale-step/2))
                    sinc = np.array([np.sinc(d*dwn) for d in opd[idx_count]])
                    sine = sine * sinc

                null_output = beam_i_null * zeta_null_i + beam_j_null * zeta_null_j -\
                2*np.sqrt(beams_ij_null) * np.sqrt(zeta_null_i * zeta_null_j) * visibility[idx_count] * sine[:,None,:]
                
                anti_null_output = beam_i_anti_null * zeta_antinull_i + beam_j_anti_null * zeta_antinull_j + \
                2*np.sqrt(beams_ij_anti_null) * np.sqrt(zeta_antinull_i * zeta_antinull_j) * visibility[idx_count] * sine[:,None,:]

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
        print('Saving data')
        if not os.path.exists(path):
            os.makedirs(path)
            
        if not activate_dark_only:
            path2 = path + 'simu_'+str(mag)+'_%04d.mat'%(bl+1)
        else:
            path2 = path + 'dark_%04d.mat'%(bl+1)
        
        date = datetime.datetime.utcnow().isoformat()
        save(np.transpose(data, axes=(0,2,1)), path2, date, DIT, na, mag, nbimg, \
             [piston_shift, piston_std, jump], [strehl_mean, strehl_std],\
             activate_dark_only, activate_turbulence, activate_phase_bias, activate_zeta, activate_oversampling, activate_opd_bias)
        
        if not os.path.isfile(save_segment_pos_path):
            print('Create mat file with segment positions')
            save_segment_positions(segment_positions, save_segment_pos_path)
    
    stop = time()
    print('Last: %.3f'%(stop-start))
    
stop0 = time()
print('Total last: %.3f'%(stop0-start0))

# =============================================================================
# Miscelleanous
# =============================================================================
output = data - data[:,:10].mean()
null = np.array([output[:,int(round(null_output_pos_i-10)):int(round(null_output_pos_i+10)),:].sum(axis=1) for null_output_pos_i in null_output_pos])
antinull = np.array([output[:,int(round(anti_null_output_pos_i-10)):int(round(anti_null_output_pos_i+10)),:].sum(axis=1) for anti_null_output_pos_i in anti_null_output_pos])
p = np.array([output[:,int(round(photo_pos_i-10)):int(round(photo_pos_i+10)),:].sum(axis=1) for photo_pos_i in photo_pos])


if activate_scan_null:
    plt.figure(figsize=(19.20,10.80))
    for i in range(6):
        scan_null = null[i,:,wl_offset:].sum(axis=1)
        scan_antinull = antinull[i,:,wl_offset:].sum(axis=1)
        plt.subplot(3,2,i+1)
        plt.plot(scan_range, scan_null, label='min@%.2f nm (%s)'%(scan_range[np.argmin(scan_null)], np.argmin(scan_null)))
        plt.plot(scan_range, scan_antinull, label='max@%.2f nm (%s)'%(scan_range[np.argmax(scan_antinull)], np.argmax(scan_antinull)))
        plt.grid()
        plt.title(null_labels[i])
        plt.legend(loc='best')
    plt.tight_layout()

 

#wl_offset = 28
#wl = wl_scale[wl_offset:]
#wl0 = np.mean(wl)
#dwl = wl.max()-wl.min()
#wl1 = wl0 + 0.5 * dwl
#dwl1 = dwl
#wl2 = wl0 - 0.5 * dwl
#dwl2 = dwl * 0.05
#x_axis = ramp * 2/wl0
#plt.figure(figsize=(19.20,10.80))
#for i in range(1):
#    scan_null = null[i,:,wl_offset:]
#    p1 = p[null_photo_map[i,0],:,wl_offset:] * zeta_minus[i, 0, wl_offset:]
#    p2 = p[null_photo_map[i,1],:,wl_offset:] * zeta_minus[i, 1, wl_offset:]
#    p1A, p1B = p1[:,(wl>=wl1-dwl1/2)&(wl<=wl1+dwl1/2)], p1[:,(wl>=wl2-dwl2/2)&(wl<=wl2+dwl2/2)]
#    p2A, p2B = p2[:,(wl>=wl1-dwl1/2)&(wl<=wl1+dwl1/2)], p2[:,(wl>=wl2-dwl2/2)&(wl<=wl2+dwl2/2)]
#    scan_nullA = scan_null[:,(wl>=wl1-dwl1/2)&(wl<=wl1+dwl1/2)]
#    scan_nullB = scan_null[:,(wl>=wl2-dwl2/2)&(wl<=wl2+dwl2/2)]
#    scan_nullA = (p1A + p2A - scan_nullA)/np.sqrt(4*p1A*p2A)
#    scan_nullB = (p1B + p2B - scan_nullB)/np.sqrt(4*p1B*p2B)
#    scan_nullA = np.nanmean(scan_nullA, axis=1)
#    scan_nullB = np.nanmean(scan_nullB, axis=1)
#    
#    ratio = scan_nullA/scan_nullB
#    expected0 = dwl2/dwl1 * np.sin(np.pi*dwl1/wl1**2*ramp) / np.sin(np.pi*dwl2/wl2**2*ramp)
#    expected = np.sin(2*np.pi/wl1 * opd[0]+phase_bias[0]) * np.sinc(dwl1/wl1**2 * opd[0])
#    expected2 = np.sin(2*np.pi/wl2 * opd[0]+phase_bias[0]) * np.sinc(dwl2/wl2**2 * opd[0])
#    plt.subplot(3,2,i+1)
##    plt.plot(x_axis, scan_nullA, '-')
##    plt.plot(x_axis, scan_nullB, '-')
#    plt.plot(x_axis, ratio, '.')
##    plt.plot(x_axis, expected, '-')
##    plt.plot(x_axis, expected2, '-')
##    plt.plot(x_axis, expected0, '.')
#    plt.plot(x_axis, ratio-expected0, '.')
#    plt.grid()
#    plt.title(null_labels[i])
#    plt.ylim(-2,2)
#plt.tight_layout()


for i in range(min(5,nbimg)):
    plt.figure()
    plt.subplot(121)
    plt.imshow(data[i], interpolation='none', aspect='auto', extent=[wl_scale[0], wl_scale[-1], detector_size[0], 0],) 
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(image_clean[i], interpolation='none', aspect='auto', extent=[wl_scale[0], wl_scale[-1], detector_size[0], 0]) 
    plt.colorbar()
    
    maxi = max(p.max(), null.max(), antinull.max())
    plt.figure(figsize=(19.20,10.80))
    for j in range(4):
        plt.subplot(4,4,j+1)
        plt.plot(wl_scale, p[j,i])
        plt.grid()
        plt.ylim(-5, maxi*1.05)
        plt.title('P%s'%(j+1))
    for j in range(4,10):
        plt.subplot(4,4,j+1)
        plt.plot(wl_scale, null[j-4,i])
        plt.grid()
        plt.ylim(-5, maxi*1.05)
        plt.title(null_labels[j-4])
    for j in range(10,16):
        plt.subplot(4,4,j+1)
        plt.plot(wl_scale, antinull[j-10,i])
        plt.grid()
        plt.ylim(-5, maxi*1.05)
        plt.title('A'+null_labels[j-10])
    plt.tight_layout()


#from matplotlib import animation
#fig = plt.figure()
#ax = plt.axes()
#time_text = ax.text(0.05, 0.01, '', transform=ax.transAxes, color='w')
#im = plt.imshow(output[0],interpolation='none')
## initialization function: plot the background of each frame
#def init():
#    im.set_data(output[0])
#    time_text.set_text('')
#    return [im] + [time_text]
## animation function.  This is called sequentially
#def animate(i):
#    im.set_array(output[i])
#    time_text.set_text('Frame %s/%s'%(i+1, image_clean.shape[0]))
#    return [im] + [time_text]
#
#anim = animation.FuncAnimation(fig, animate, init_func=init, frames=output.shape[0], interval=200, blit=True)

#plt.figure()
#plt.plot(wl_scale, p[0,0]/np.max(p[0,0]), label='p1')
#plt.plot(wl_scale, p[1,0]/np.max(p[1,0]), label='p2')
#plt.plot(wl_scale, null[0,0]/np.max(null[0,0]), label='n1')
#plt.plot(wl_scale, antinull[0,0]/np.max(antinull[0,0]), label='an1')
#plt.grid()
#plt.legend(loc='best')
    
