# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 16:37:34 2019

@author: mamartinod
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import glint_simulator_classes as classes
from glint_simulator_functions import gaus, setZetaCoeff, save, createObject, skewedGaussian
from itertools import combinations
import os
import datetime
from timeit import default_timer as time

# =============================================================================
# Settings
# =============================================================================
activate_dark_only = False
activate_scan_null = False
activate_turbulence_injection = True
activate_turbulence_piston = True
activate_advanced_turb = False
activate_phase_bias = False
activate_opd_bias = True
activate_zeta = True
#activate_transmission = False
activate_spectrum = False
activate_oversampling = True
activate_crosstalk = False
select_noise = [True, True] # Dark and Readout noises
conversion = True
active_beams = np.array([1,1,1,1], dtype=np.bool)
nb_block = (0, 1000)
nbimg = 100
save_data = True
path = '/mnt/96980F95980F72D3/glint_data/simu_ron_regime/'
namefile = 'pure_injection'
auto_talk_beams = [0, 1]
scanned_segment = 0
scan_bounds = (-2500, 2500) # In nm

# =============================================================================
# Fundamental constants
# =============================================================================
c = 3.0E+8 #m/s
h = 6.63E-34 #J.s
flux_ref = 1.26E-12 #Reference flux in W/m^2/nm
wl0 = 1557#1602
opd0 = wl0/4.

# =============================================================================
# Spectral band
# =============================================================================
wl_min, wl_max = 1357, 1832 # In nm
bandwidth = wl_max - wl_min

#==============================================================================
# Properties of the detector
#==============================================================================
DIT = 1/1394 # in sec
Ndark = 1500 # in e-/px/s
Ndark *= DIT # in e-/px/fr
sigma_ron = 30. # in e-/px/fr
wl_stop = 1660

detector_size = (344, 96) #resp. position axes and wavelength

channel_width = bandwidth / (detector_size[1])

QE = 0.85
gainsys = 0.5 # ADU/electron
offset = 2750

step = 5 #In nm
wl_scale = np.arange(wl_min, wl_max+5,5, dtype=np.float64) #In nm
wl_scale = wl_scale[::-1]
wl_offset = np.where(wl_scale>wl_stop)[0][-1]+1 # in pixel, shift of the bandwidth on the detecotr hence a blank in a part of it

wave_number = 1/wl_scale # In microns^-1
position_scale = np.arange(0, detector_size[0]) #In pixel

#==============================================================================
# Properties of the pupils and the instrument
#==============================================================================
nb_pupils = 4 # Number of pupils
liste_pupils = np.arange(nb_pupils) # Id of the pupils
diam = 1.075 #in meter
central_obs = 0.
surface = np.pi * (diam**2-central_obs**2) / 4 #en m^2

# =============================================================================
# Properties of the integrated optics and injection
# =============================================================================
beam_comb = [str(i)+''+str(j) for i, j in combinations(np.arange(4), 2)]

rho0 = 0.8

segment_positions = np.zeros(4,)
segment_positions[0] = 830.
segment_positions[2] = 1055.

# Order of pairs of beams: 12, 31, 14, 23, 42, 34 (N1, N5, N3, N2, N6, N4)
if activate_opd_bias:
    opd_bias = np.zeros(6)
    opd_bias[0] = -3.28469298e+03 + 2000 # N1
    opd_bias[1] = -9.25630712e+03 + 2000 # N5
    opd_bias[2] = -15000 + 2000 # N3
    opd_bias[3] = -15000 + 2000 # N2
    opd_bias[4] = -1.02965460e+04 + 2000 # N6
    opd_bias[5] = -3.73491385e+03 + 2000 # N4
else:
    print('no opd bias')
    opd_bias = np.ones(6,) * opd0
    opd_bias[1] = -opd_bias[3] - opd_bias[0]  # 31 = 32 + 21 = -23 - 12 i.e. N5
    opd_bias[2] = -opd_bias[1] + opd_bias[5] # 14 = 13 + 34 = -31 + 34 i.e. N3
    opd_bias[4] = -opd_bias[5] - opd_bias[3] # 42 = 43 + 32 = -34 - 23 i.e. N6
    segment_positions = np.zeros(4,)

if activate_phase_bias:
    phase_bias = np.zeros(6)
    phase_bias[0] = 0.
    phase_bias[1] = 0.
    phase_bias[2] = 0.
    phase_bias[3] = 0.
    phase_bias[4] = 0.
    phase_bias[5] = 0.
else:
    phase_bias = np.zeros(6,)
    


# =============================================================================
# Properties of atmosphere - Gaussian model
# =============================================================================
piston_mu = 0
piston_shift, jump = wl0/2, 0.
piston_std = 0.
strehl_mean, strehl_std = 0.5, 0.5
r0 = 0.3 # @ 500 nm, in meter, roughly sr=0.9..0.95 at 1.5 microns
wl_wfs = 500e-9 # wavelength of the wavefront sensor
wind_speed = 10 # wind speed in m/s

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
baselines = np.array([5.55, 3.2, 4.65, 6.45, 5.68, 2.15])

#==============================================================================
# Stars
#==============================================================================
mag = -1.5
ud_diam = 20. # in mas
if ud_diam != 0:
    visibility, baselines = createObject('ud', ud_diam, wl0*1e-9)
    visibility = np.array([visibility[0], visibility[4], visibility[2], visibility[1], visibility[5], visibility[3]])
else:
    visibility = np.ones(6)
na = (1-visibility) / (1+visibility)

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
    zeta_minus, zeta_plus = setZetaCoeff(wl_scale, 'zeta_coeff.hdf5', wl_stop, save=False, plot=False)
else:
    zeta_minus = np.ones((6,2,96))*0.5
    zeta_plus = np.ones((6,2,96))*0.5
                      
transmission = np.array([1/(1 + zeta_minus[0,0] + zeta_plus[0,0] + zeta_minus[1,0] + zeta_plus[1,0] + zeta_minus[2,0] + zeta_plus[2,0]),
                         1/(1 + zeta_minus[0,1] + zeta_plus[0,1] + zeta_minus[3,0] + zeta_plus[3,0] + zeta_minus[4,0] + zeta_plus[4,0]),
                         1/(1 + zeta_minus[1,1] + zeta_plus[1,1] + zeta_minus[3,1] + zeta_plus[3,1] + zeta_minus[5,0] + zeta_plus[5,0]),
                         1/(1 + zeta_minus[2,1] + zeta_plus[2,1] + zeta_minus[4,1] + zeta_plus[4,1] + zeta_minus[5,1] + zeta_plus[5,1])])
transmission[~active_beams] = 0
transmission[transmission==1] = 0

if activate_scan_null:
    scan_range = np.linspace(scan_bounds[0], scan_bounds[1], 1001)
    nbimg = scan_range.size
    activate_turbulence_injection = False
    activate_turbulence_piston = False
    activate_dark_only = False
    select_noise = [False, False] # Dark and Readout noises
    conversion = False
    nb_block = (0, 1)
    save_data = False
    mag = -5
    visibility[:] = 1.
    na[:] = 0.
    
liste_iminus = []
liste_iplus = []
liste_photo = []
liste_avg = []
control_avg = []

start0 = time()
for bl in range(*nb_block):
    start = time()
    print("Generating block %s / %s"%(bl+1,nb_block[1]))
    if activate_scan_null:
        segment_positions = np.zeros((4, scan_range.size))
        segment_positions[scanned_segment] = scan_range
        
    if activate_advanced_turb:
        print('Generating turbulence.')
        turb = classes.goas(wl0*1e-9, nbimg)
        diff_phases, strehl = turb.run(r0, wl_wfs, 1)
        diff_phases = diff_phases * wl0/(2*np.pi)
        print('Turbulence generated.')
        del turb
        if not activate_turbulence_injection:
            strehl = np.ones((nb_pupils, nbimg))
        if not activate_turbulence_piston:
            diff_phases = np.zeros((6, nbimg))
    else:
        if activate_turbulence_injection:
            strehl = np.random.normal(strehl_mean, strehl_std, size=(nb_pupils, nbimg))
            while np.min(strehl) < 0 or np.max(strehl) > 1:
                idx = np.where((strehl<0)|(strehl>1))
                strehl[idx] = np.random.normal(strehl_mean, strehl_std, size=idx[0].size)
        else:
            strehl = np.ones((nb_pupils, nbimg))
        
        if activate_turbulence_piston:
            piston_pupils = np.random.normal(piston_mu, piston_std, (4, nbimg))
            diff_phases = np.array([piston_pupils[0]-piston_pupils[1],
                                    piston_pupils[1]-piston_pupils[2],
                                    piston_pupils[0]-piston_pupils[3],
                                    piston_pupils[2]-piston_pupils[3],
                                    piston_pupils[2]-piston_pupils[1],
                                    piston_pupils[3]-piston_pupils[1]])
        else:
            diff_phases = np.zeros((6, nbimg))

    # strehl[0] = np.linspace(0.1, 1., nbimg)
    # np.savetxt(path+'strehl.txt', strehl)
    rho = rho0 * strehl
    opd = np.ones((int(comb(nb_pupils, 2)), nbimg))
    
    # Order of pairs of beams: 12, 31, 14, 23, 42, 34 (N1, N5, N3, N2, N6, N4)
    # Independant pairs: N1: 12, N2: 23, N4: 34
    opd[0] = opd_bias[0] + diff_phases[0] + 2*(segment_positions[0] - segment_positions[1]) # N1: 12
    opd[3] = opd_bias[3] + diff_phases[1] + 2*(segment_positions[1] - segment_positions[2]) # N2: 23
    opd[5] = opd_bias[5] + diff_phases[3] + 2*(segment_positions[2] - segment_positions[3]) # N4: 34
    
    opd[1] = opd_bias[1] + diff_phases[4] + 2*(segment_positions[2] - segment_positions[0])# 31 = 32 + 21 = -23 - 12 i.e. N5
    opd[2] = opd_bias[2] + diff_phases[2] + 2*(segment_positions[0] - segment_positions[3])# 14 = 13 + 34 = -31 + 34 i.e. N3
    opd[4] = opd_bias[4] + diff_phases[5] + 2*(segment_positions[3] - segment_positions[1])# 42 = 43 + 32 = -34 - 23 i.e. N6

    opd_ramp = np.linspace(-wl0-opd[0,0], wl0-opd[0,0],nbimg, endpoint=False)
    # opd[0] += opd_ramp
    # np.savetxt(path+'opd.txt', opd_ramp)
    
    image_clean = np.zeros((nbimg, detector_size[0], detector_size[1]))
    
    if not activate_dark_only:
        idx_count = 0
        if activate_spectrum:
            flat_spectrum = np.ones(detector_size[1])
            flat_spectrum[:wl_offset] = 0.
            spectrum = skewedGaussian(wl_scale, 1., wl_scale.mean(), 50, -1)
            spectrum = spectrum / np.sum(spectrum)
            wl_offset = 0
            plt.figure()
            plt.plot(wl_scale, spectrum, lw=3)
            plt.grid()
            plt.xticks(size=25);plt.yticks(size=25)
            plt.title('Spectrum', size=35)
            plt.ylabel('Normalised intensity', size=30)
            plt.xlabel('Wavelength (nm)', size=30)
            plt.tight_layout()
        else:
            spectrum = np.ones(detector_size[1])
            spectrum[:wl_offset] = 0.
            spectrum = spectrum / spectrum.sum()
            
        for i in liste_pupils:
            N = QE * surface * channel_width * DIT * np.power(10, -0.4*mag) * flux_ref * wl_scale.mean()*1.E-9 /(h*c) * spectrum
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
    
    # =============================================================================
    # Noising frames
    # =============================================================================
    data_noisy = classes.Noising(nbimg, image_clean)
    data_noisy.addNoise(Ndark, gainsys, offset, sigma_ron, active=select_noise, convert=conversion)
    data = data_noisy.output
    
    avg = data[:,:,:20].mean()
    output = data - avg
    p = np.array([output[:,int(round(photo_pos_i)):int(round(photo_pos_i+1)),:].sum(axis=1) for photo_pos_i in photo_pos])
    p2 = np.array([image_clean[:,int(round(photo_pos_i)):int(round(photo_pos_i+1)),:].sum(axis=1) for photo_pos_i in photo_pos])
    liste_photo.append(p)
    Iminus = np.array([output[:,int(round(null_output_pos_i)):int(round(null_output_pos_i+1)),:].sum(axis=1) for null_output_pos_i in null_output_pos])
    Iplus = np.array([output[:,int(round(anti_null_output_pos_i)):int(round(anti_null_output_pos_i+1)),:].sum(axis=1) for anti_null_output_pos_i in anti_null_output_pos])
    liste_iminus.append([Iminus[0], Iminus[3], Iminus[2], Iminus[5], Iminus[1], Iminus[4]])
    liste_iplus.append([Iplus[0], Iplus[3], Iplus[2], Iplus[5], Iplus[1], Iplus[4]])
    liste_avg.append(avg)
    control_avg.append(output[:,:20,:].mean())
    
    # =============================================================================
    # Save data
    # =============================================================================
    if save_data:
        print('Saving data')
        if not os.path.exists(path):
            os.makedirs(path)
            
        if not activate_dark_only:
            path2 = path + namefile +'_%04d.mat'%(bl+1)
        else:
            path2 = path + 'dark_%04d.mat'%(bl+1)
        
        date = datetime.datetime.utcnow().isoformat()
        # np.savetxt(path+'opd_range_%04d.txt'%(bl+1), opd)
        save(np.transpose(data, axes=(0,2,1)), path2, date, DIT, na, mag, nbimg, \
             [piston_shift, piston_std, jump], [strehl_mean, strehl_std],\
             activate_dark_only, activate_turbulence_injection, activate_turbulence_piston, 
             activate_phase_bias, activate_zeta, activate_oversampling, activate_opd_bias, ud_diam)
    
    stop = time()
    print('Last: %.3f'%(stop-start))
    
stop0 = time()
print('Total last: %.3f'%(stop0-start0))

liste_photo = np.array(liste_photo)
liste_iminus = np.array(liste_iminus)
liste_iplus = np.array(liste_iplus)

# =============================================================================
# Scan of fringes
# =============================================================================
output = data - data[:,:,:10].mean()
Iminus = np.array([output[:,int(round(null_output_pos_i)):int(round(null_output_pos_i+1)),:].sum(axis=1) for null_output_pos_i in null_output_pos])
Iplus = np.array([output[:,int(round(anti_null_output_pos_i)):int(round(anti_null_output_pos_i+1)),:].sum(axis=1) for anti_null_output_pos_i in anti_null_output_pos])
p = np.array([output[:,int(round(photo_pos_i)):int(round(photo_pos_i+1)),:].sum(axis=1) for photo_pos_i in photo_pos])
null = Iminus[:,:,(wl_scale>=1400)&(wl_scale<=1650)].sum(axis=-1) / Iplus[:,:,(wl_scale>=1400)&(wl_scale<=1650)].sum(axis=-1)
['N1', 'N5', 'N3', 'N2', 'N6', 'N4']
null = np.array([null[0], null[3], null[2], null[5], null[1], null[4]])

if activate_scan_null:
    plt.figure(figsize=(19.20,10.80))
    for i in range(6):
        scan_null = Iminus[i,:,wl_offset:].sum(axis=1)
        scan_antinull = Iplus[i,:,wl_offset:].sum(axis=1)
        plt.subplot(3,2,i+1)
        plt.plot(scan_range, scan_null, label='min@%.2f nm (%s)'%(scan_range[np.argmin(scan_null)], np.argmin(scan_null)))
        plt.plot(scan_range, scan_antinull, label='max@%.2f nm (%s)'%(scan_range[np.argmax(scan_antinull)], np.argmax(scan_antinull)))
        plt.grid()
        plt.title(null_labels[i])
        plt.legend(loc='best')
    plt.tight_layout()

    plt.figure(figsize=(19.20,10.80))
    for i in range(6):
        scan_null = Iminus[i,:,wl_offset:].sum(axis=1)
        scan_antinull = Iplus[i,:,wl_offset:].sum(axis=1)
        plt.subplot(3,2,i+1)
        plt.plot(scan_range, scan_null/scan_antinull, label='Ratio')
        plt.grid()
        plt.title(null_labels[i])
        plt.legend(loc='best')
    plt.tight_layout() 

# =============================================================================
# Miscelleanous
# =============================================================================
# liste_photo = liste_photo[:,:,:,(wl_scale>=1400)&(wl_scale<=1650)].sum(axis=-1)
# liste_iminus = liste_iminus[:,:,:,(wl_scale>=1400)&(wl_scale<=1650)].sum(axis=-1)
# liste_iplus = liste_iplus[:,:,:,(wl_scale>=1400)&(wl_scale<=1650)].sum(axis=-1)

pA = liste_photo[:,0]
pB = liste_photo[:,1]
destructif = liste_iminus[:,0]
constructif = liste_iplus[:,0]
ddm = opd_bias[0] + 2*(segment_positions[0] - segment_positions[1]) # N1: 12
reconstruit_plus = pA * zeta_plus[0,0] + pB * zeta_plus[0,1] + \
                    2 * (pA * pB)**0.5 * (zeta_plus[0,0] * zeta_plus[0,1])**0.5 * np.sin(2*np.pi/wl_scale*ddm)
reconstruit_moins = pA * zeta_minus[5,0] + pB * zeta_minus[5,1] + \
                    2 * (pA * pB)**0.5 * (zeta_minus[0,0] * zeta_minus[0,1])**0.5 * np.sin(2*np.pi/wl_scale*ddm)

destructif = destructif[:,:,(wl_scale>=1525)&(wl_scale<=1575)].sum(axis=-1)
constructif = constructif[:,:,(wl_scale>=1525)&(wl_scale<=1575)].sum(axis=-1)
reconstruit_plus = reconstruit_plus[:,:,(wl_scale>=1525)&(wl_scale<=1575)].sum(axis=-1)
reconstruit_moins = reconstruit_moins[:,:,(wl_scale>=1525)&(wl_scale<=1575)].sum(axis=-1)

destructif = destructif.reshape(-1)
constructif = constructif.reshape(-1)
reconstruit_plus = reconstruit_plus.reshape(-1)
reconstruit_moins = reconstruit_moins.reshape(-1)

destructif = destructif[~np.isnan(destructif)]
constructif = constructif[~np.isnan(constructif)]
reconstruit_plus = reconstruit_plus[~np.isnan(reconstruit_plus)]
reconstruit_moins = destructif[~np.isnan(reconstruit_moins)]

histod = np.histogram(destructif, int(destructif.size**0.5), density=True)
histoc = np.histogram(constructif, int(constructif.size**0.5), density=True)
histord = np.histogram(reconstruit_moins, int(reconstruit_moins.size**0.5), density=True)
historc = np.histogram(reconstruit_plus, int(reconstruit_plus.size**0.5), density=True)

plt.figure()
plt.plot(histod[1][:-1], histod[0], '.', label='I-')
plt.plot(histord[1][:-1], histord[0], '-', label='I- recons')
plt.plot(histoc[1][:-1], histoc[0], '.', label='I+')
plt.plot(historc[1][:-1], historc[0], '-', label='I+ recons')
plt.legend(loc='best')
print(reconstruit_plus.mean()/constructif.mean())
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

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=output.shape[0], interval=10, blit=True)
#
#histo = np.histogram(2*np.pi/wl0*opd[0], int(opd[0].size**0.5), density=True)
#plt.figure()
#plt.plot(histo[1][:-1], histo[0])
#plt.grid()
#plt.title('Histogram of OPD (N1)')
#
#plt.figure()
#for i in range(6):
#    plt.subplot(3,2,i+1)
#    plt.plot(null[i])
#    plt.grid()
#    plt.title('Null %s'%(i+1))

    
# # =============================================================================
# # Fringe tracking
# # =============================================================================
# def binning(arr, binning, axis=0, avg=False):
#     """
#     Bin frames together
    
#     :Parameters:
#         **arr**: nd-array
#             Array containing data to bin
#         **binning**: int
#             Number of frames to bin
#         **axis**: int
#             axis along which the frames are
#         **avg**: bol
#             If ``True``, the method returns the average of the binned frame.
#             Otherwise, return its sum.
            
#     :Attributes:
#         Change the attributes
        
#         **data**: ndarray 
#             datacube
#     """
#     if binning is None:
#         binning = arr.shape[axis]
        
#     shape = arr.shape
#     crop = shape[axis]//binning*binning # Number of frames which can be binned respect to the input value
#     arr = np.take(arr, np.arange(crop), axis=axis)
#     shape = arr.shape
#     if axis < 0:
#         axis += arr.ndim
#     shape = shape[:axis] + (-1, binning) + shape[axis+1:]
#     arr = arr.reshape(shape)
#     if not avg:
#         arr = arr.sum(axis=axis+1)
#     else:
#         arr = arr.mean(axis=axis+1)
    
#     return arr

# wl_min = 1450
# wl_max = 1650
# wl = wl_scale[(wl_scale>=wl_min)&(wl_scale<=wl_max)]
# Iminus = Iminus[:,:,(wl_scale>=wl_min)&(wl_scale<=wl_max)]
# Iplus = Iplus[:,:,(wl_scale>=wl_min)&(wl_scale<=wl_max)]
# nb_spec_chan = 2
# iminus = Iminus[0]
# iplus = Iplus[0]

# iminus4 = binning(iminus, iminus.shape[1]//nb_spec_chan, 1, True)[::-1]
# iplus4 = binning(iplus, iplus.shape[1]//nb_spec_chan, 1, True)[::-1]
# wl_4channels = binning(wl, wl.size//nb_spec_chan, 0, True)[::-1]

# intranull = np.array([(iminus4[:,0] - iminus4[:,-1])/(iminus4[:,0]+iminus4[:,-1]), (iplus4[:,0] - iplus4[:,-1])/(iplus4[:,0]+iplus4[:,-1])])
# visi = (iplus4-iminus4)/(iplus4+iminus4)

# # np.savez(path+'opd_zeta_nospectrum.npz', intranull=intranull, visi=visi, strehl=strehl[0], opd=opd_ramp)
# a = np.load(path+'injection_nozeta_nospectrum.npz')

# plt.figure(figsize=(19.2, 10.8))
# plt.plot(strehl[0], intranull[0], '.', markersize=10, label=r'$\frac{I^{-}_{%s} - I^{-}_{%s}}{I^{-}_{%s} + I^{-}_{%s}}$ (Skewed spectrum)'%(*np.round(wl_4channels), *np.round(wl_4channels)))
# plt.plot(strehl[0], intranull[1], '.', markersize=10, label=r'$\frac{I^{+}_{%s} - I^{+}_{%s}}{I^{+}_{%s} + I^{+}_{%s}}$ (Skewed spectrum)'%(*np.round(wl_4channels), *np.round(wl_4channels)))
# plt.gca().set_prop_cycle(None)
# plt.plot(a['strehl'], a['intranull'][0], '+', markersize=10, label=r'$\frac{I^{-}_{%s} - I^{-}_{%s}}{I^{-}_{%s} + I^{-}_{%s}}$ (Flat spectrum)'%(*np.round(wl_4channels), *np.round(wl_4channels)))
# plt.plot(a['strehl'], a['intranull'][1], '+', markersize=10, label=r'$\frac{I^{+}_{%s} - I^{+}_{%s}}{I^{+}_{%s} + I^{+}_{%s}}$ (Flat spectrum)'%(*np.round(wl_4channels), *np.round(wl_4channels)))
# plt.grid()
# plt.xlim(0)
# plt.ylim(-1.05,1.2)
# plt.legend(loc='best', fontsize=25, ncol=2)
# plt.xticks(size=30);plt.yticks(size=30)
# plt.xlabel('Strehl ratio', size=35)
# plt.ylabel('Amplitude (AU)', size=35)
# plt.tight_layout()

# plt.figure(figsize=(19.2, 10.8))
# plt.plot(opd_ramp, intranull[0], '.', markersize=10, label=r'$\frac{I^{-}_{%s} - I^{-}_{%s}}{I^{-}_{%s} + I^{-}_{%s}}$ (Skewed spectrum)'%(*np.round(wl_4channels), *np.round(wl_4channels)))
# plt.plot(opd_ramp, intranull[1], '.', markersize=10, label=r'$\frac{I^{+}_{%s} - I^{+}_{%s}}{I^{+}_{%s} + I^{+}_{%s}}$ (Skewed spectrum)'%(*np.round(wl_4channels), *np.round(wl_4channels)))
# plt.gca().set_prop_cycle(None)
# plt.plot(a['opd'], a['intranull'][0], '+', markersize=10, label=r'$\frac{I^{-}_{%s} - I^{-}_{%s}}{I^{-}_{%s} + I^{-}_{%s}}$ (Flat spectrum)'%(*np.round(wl_4channels), *np.round(wl_4channels)))
# plt.plot(a['opd'], a['intranull'][1], '+', markersize=10, label=r'$\frac{I^{+}_{%s} - I^{+}_{%s}}{I^{+}_{%s} + I^{+}_{%s}}$ (Flat spectrum)'%(*np.round(wl_4channels), *np.round(wl_4channels)))
# plt.grid()
# plt.ylim(-1,2)
# plt.legend(loc='best', fontsize=30, ncol=2)
# plt.xticks(size=30);plt.yticks(size=30)
# plt.xlabel('Gap between current OPD and the nominal one (in nm)', size=35)
# plt.ylabel('Amplitude (AU)', size=35)
# # plt.ylim(-1,1)
# plt.tight_layout()
    
# plt.figure(figsize=(19.2, 10.8))
# plt.plot(strehl[0], visi[:,0]/visi[:,-1], '.', markersize=10, label=r'$\frac{V_{%s}}{V_{%s}}$ (Skewed spectrum)'%(np.round(wl_4channels[0]), np.round(wl_4channels[-1])))
# plt.plot(a['strehl'], a['visi'][:,0]/a['visi'][:,-1], '.', markersize=10, label=r'$\frac{V_{%s}}{V_{%s}}$ (Flat spectrum)'%(np.round(wl_4channels[0]), np.round(wl_4channels[-1])))
# plt.grid()
# plt.xlim(0)
# plt.ylim(-1.05, 1.2)
# plt.legend(loc='best', fontsize=30)
# plt.xticks(size=30);plt.yticks(size=30)
# plt.xlabel('Strehl ratio', size=35)
# plt.ylabel('Amplitude (AU)', size=35)
# plt.tight_layout()

# plt.figure(figsize=(19.2, 10.8))
# plt.plot(opd_ramp, visi[:,0]/visi[:,-1], '.', markersize=10, label=r'$\frac{V_{%s}}{V_{%s}}$ (Skewed spectrum)'%(np.round(wl_4channels[0]), np.round(wl_4channels[-1])))
# plt.plot(a['opd'], a['visi'][:,0]/a['visi'][:,-1], '.', markersize=10, label=r'$\frac{V_{%s}}{V_{%s}}$ (Flat spectrum)'%(np.round(wl_4channels[0]), np.round(wl_4channels[-1])))
# plt.grid()
# plt.ylim(-1, 2)
# plt.legend(loc='best', fontsize=30)
# plt.xticks(size=30);plt.yticks(size=30)
# plt.xlabel('Gap between current OPD and the nominal one (in nm)', size=35)
# plt.ylabel('Amplitude (AU)', size=35)
# # plt.ylim(-1,1)
# plt.tight_layout()


# # plt.figure(figsize=(19.2, 10.8))
# # plt.plot(strehl[0], intranull[1]/a['intranull'][1], '.', markersize=10)
# # plt.grid()
# # plt.xlim(0)
# # # plt.ylim(-0.05,1.05)
# # plt.legend(loc='best', fontsize=30)
# # plt.xticks(size=30);plt.yticks(size=30)
# # plt.xlabel('Strehl ratio', size=35)
# # plt.ylabel('Amplitude (AU)', size=35)
# # plt.tight_layout()
