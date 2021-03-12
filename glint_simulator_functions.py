#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:15:44 2019

@author: mam
"""

import numpy as np
import cupy as cp
from cupyx.scipy.special.statistics import ndtr
import matplotlib.pyplot as plt
import h5py
import os
from scipy.io import savemat
import scipy.special as sp
from scipy.interpolate import interp1d

def doubleGaussCdf(x, mu1, mu2, sig, A):
    return 1/(1+A) * ndtr((x-mu1)/(sig)) + A/(1+A) * ndtr((x-mu2)/(sig))

def rv_generator(absc, cdf, nsamp):
    '''
    absc : cupy-array, x-axis of the histogram
    cdf : cupy-array, normalized arbitrary pdf to use to generate rv
    nsamp : int, number of samples to draw
    '''

    cdf, mask = cp.unique(cdf, True)    
    cdf_absc = absc[mask]

    rv_uniform = cp.random.rand(nsamp, dtype=cp.float32)
    output_samples = cp.zeros(rv_uniform.shape, dtype=cp.float32)
    interpolate_kernel(rv_uniform, cdf, cdf.size, cdf_absc, output_samples)
    
    return output_samples

interpolate_kernel = cp.ElementwiseKernel(
    'float32 x_new, raw float32 xp, int32 xp_size, raw float32 yp', 
    'raw float32 y_new',
    
    '''  
    int high = xp_size - 1;
    int low = 0;
    int mid = 0;
    
    while(high - low > 1)
    {
        mid = (high + low)/2;
        
        if (xp[mid] <= x_new)
        {
            low = mid;
        }
        else
        {
            high = mid;
        }
    }
    y_new[i] = yp[low] + (x_new - xp[low])  * (yp[low+1] - yp[low]) / (xp[low+1] - xp[low]);

    if (x_new < xp[0])
    {
         y_new[i] = yp[0];
    }
    else if (x_new > xp[xp_size-1])
    {
         y_new[i] = yp[xp_size-1];
    }
        
    '''
    )

    
def gaus(x, mu, sigma):
    y = np.exp(-(x - mu)**2/(2 * sigma**2))
    y /= y.sum()
    return y

def chromatic_splitters(ratios, wl_scale, wl0, slope):
    chroma_ratios = np.zeros((ratios.shape[0], 2, wl_scale.size))
    chromatism = slope * (wl_scale - wl0)
    chroma_ratios[:,0] = ratios[:,0,None] + chromatism[None,:]
    chroma_ratios[:,1] = ratios[:,0,None] + chromatism[None,:]
        
    return chroma_ratios

def save(arr, path, date, DIT, na, mag, nbimg, piston, strehl, dark_only_switch, \
         activate_turbulence_injection, activate_turbulence_piston, activate_phase_bias, activate_zeta, activate_oversampling, activate_opd_bias, ud_diam):
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
        f.attrs['piston'] = np.array(piston)
        f.attrs['strehl'] = np.array(strehl)
        f.attrs['obj size'] = ud_diam
        f.attrs['dark_only_switch'] = dark_only_switch
        f.attrs['activate_turbulence_injection'] = activate_turbulence_injection
        f.attrs['activate_turbulence_piston'] = activate_turbulence_piston
        f.attrs['activate_phase_bias'] = activate_phase_bias
        f.attrs['activate_zeta'] = activate_zeta
        f.attrs['activate_oversampling'] = activate_oversampling
        f.attrs['activate_opd_bias'] = activate_opd_bias
        f.create_dataset('imagedata', data=arr)

def setZetaCoeff(wl_scale, path, wl_stop, save, plot):
    with h5py.File(path, 'r') as zeta0:
        wl = np.array(zeta0['wl_scale'])
        
        zeta_minus0 = np.array([[zeta0['b1null1'], zeta0['b2null1']],
                      [zeta0['b1null5'], zeta0['b3null5']],
                      [zeta0['b1null3'], zeta0['b4null3']],
                      [zeta0['b2null2'], zeta0['b3null2']],
                      [zeta0['b2null6'], zeta0['b4null6']],
                      [zeta0['b3null4'], zeta0['b4null4']]])
        zeta_plus0 = np.array([[zeta0['b1null7'], zeta0['b2null7']],
                      [zeta0['b1null11'], zeta0['b3null11']],
                      [zeta0['b1null9'], zeta0['b4null9']],
                      [zeta0['b2null8'], zeta0['b3null8']],
                      [zeta0['b2null12'], zeta0['b4null12']],
                      [zeta0['b3null10'], zeta0['b4null10']]])
        
        zeta_minus = np.array([[np.interp(wl_scale[::-1], wl[::-1][(wl[::-1]<=wl_stop)], selt[::-1][(wl[::-1]<=wl_stop)], left=0, right=0) for selt in elt] for elt in zeta_minus0])
        zeta_plus = np.array([[np.interp(wl_scale[::-1], wl[::-1][(wl[::-1]<=wl_stop)], selt[::-1][(wl[::-1]<=wl_stop)], left=0, right=0) for selt in elt] for elt in zeta_plus0])
        
        zeta_minus = zeta_minus[:,:,::-1]
        zeta_plus = zeta_plus[:,:,::-1]
        
        zeta_minus[zeta_minus<0] = 0
        zeta_plus[zeta_plus<0] = 0
        
        if plot:
            plt.figure()
            plt.plot(wl, zeta0['b1null1'], 'o-')
            plt.plot(wl_scale, zeta_minus[0,0], 'o-')
            plt.grid()
        
    if save:
        with h5py.File('/mnt/96980F95980F72D3/glint/simulation/zeta_coeff_simu.hdf5', 'w') as newzeta:
            newzeta.create_dataset('wl_scale', data=wl_scale)
            
            newzeta.create_dataset('b1null1', data=zeta_minus[0,0])
            newzeta.create_dataset('b2null1', data=zeta_minus[0,1])
            newzeta.create_dataset('b1null5', data=zeta_minus[1,0])
            newzeta.create_dataset('b3null5', data=zeta_minus[1,1])
            newzeta.create_dataset('b1null3', data=zeta_minus[2,0])
            newzeta.create_dataset('b4null3', data=zeta_minus[2,1])       
            newzeta.create_dataset('b2null2', data=zeta_minus[3,0])
            newzeta.create_dataset('b3null2', data=zeta_minus[3,1])  
            newzeta.create_dataset('b2null6', data=zeta_minus[4,0])
            newzeta.create_dataset('b4null6', data=zeta_minus[4,1])  
            newzeta.create_dataset('b3null4', data=zeta_minus[5,0])
            newzeta.create_dataset('b4null4', data=zeta_minus[5,1])

            newzeta.create_dataset('b1null7', data=zeta_plus[0,0])
            newzeta.create_dataset('b2null7', data=zeta_plus[0,1])
            newzeta.create_dataset('b1null11', data=zeta_plus[1,0])
            newzeta.create_dataset('b3null11', data=zeta_plus[1,1])
            newzeta.create_dataset('b1null9', data=zeta_plus[2,0])
            newzeta.create_dataset('b4null9', data=zeta_plus[2,1])       
            newzeta.create_dataset('b2null8', data=zeta_plus[3,0])
            newzeta.create_dataset('b3null8', data=zeta_plus[3,1])  
            newzeta.create_dataset('b2null12', data=zeta_plus[4,0])
            newzeta.create_dataset('b4null12', data=zeta_plus[4,1])  
            newzeta.create_dataset('b3null10', data=zeta_plus[5,0])
            newzeta.create_dataset('b4null10', data=zeta_plus[5,1])            
            
    return zeta_minus, zeta_plus

def rv_gen_doubleGauss(shape, mu1, mu2, sig1, A):
    nsamp = np.prod(shape)
    x, step = cp.linspace(-2500,2500, 10000, endpoint=False, retstep=True, dtype=cp.float32)
    cdf = doubleGaussCdf(x, mu1, mu2, sig1, A)
    cdf = cp.asarray(cdf, dtype=cp.float32)
    rv = cp.asnumpy(rv_generator(x, cdf, nsamp))
    rv = np.reshape(rv, shape)
    return rv

def save_segment_positions(segment_positions, path):
    mat_dic = {}
    mat_dic['PTTPositionFlat'] = np.zeros((37,3))
    mat_dic['PTTPositionOff'] = np.zeros((37,3))
    mat_dic['PTTPositionOn'] = np.zeros((37,3))
    mat_dic['PTTPositionOn'][28,0] = segment_positions[0]/1000 # Convert to microns
    mat_dic['PTTPositionOn'][34,0] = segment_positions[1]/1000
    mat_dic['PTTPositionOn'][25,0] = segment_positions[2]/1000
    mat_dic['PTTPositionOn'][23,0] = segment_positions[3]/1000
    
    savemat(path, mat_dic)
    

def createObject(kind, *args):
    """
    Wrapper for the creation of different kind of objects set by the **kind** parameter.
    See ``notes`` for more details.
    
    :Parameters:
        **kind: str**
            kind of object to simulate: ud = uniform disk, binary = binary system of uniform disks
            
    :Returns:
        the outputs of the called function.
        
    :**Notes**:
        ``kind`` accept the following entries:
            * ``ud`` which parameters are:
                **angle: float** 
                    angular diameter of the source in mas
                **base: float**
                    length of the baseline, in meter
                **lamb: float**
                    observing wavelength, in meter
                
                It returns the visibility.
                
            * ``binary`` which parameters are:
                **diam1, diam2: floats**
                    angular diameters of both components, in mas
                **F1, F2: floats**
                    bolometric flux of both components
                **separation: float**
                    angular separation between the components, in mas
                **angular_position: float**
                    angular position of the component in degree
                **u, v: floats**
                    uv-plane coordinates of the baselines
                **lamb: float**
                    wavelength in meter
                
                It returns the tuple of modulus of the visibility and the phase.
    """
    x_mcoord = np.array([2.725, -2.812, -2.469, -0.502]) # x-coordinates of N telescopes in meter
    y_mcoord = np.array([2.317, 1.685, -1.496, -2.363]) # y-coordinates of N telescopes in meter
    baseline1 = np.array([x_mcoord[1]-x_mcoord[0], y_mcoord[1]-y_mcoord[0]]) # Null1 (AD)
    baseline2 = np.array([x_mcoord[2]-x_mcoord[0], y_mcoord[2]-y_mcoord[0]]) # Null2 (AC)
    baseline3 = np.array([x_mcoord[1]-x_mcoord[3], y_mcoord[1]-y_mcoord[3]]) # Null3 (BD)
    baseline4 = np.array([x_mcoord[3]-x_mcoord[2], y_mcoord[3]-y_mcoord[2]]) # Null4 (CB)
    baseline5 = np.array([x_mcoord[2]-x_mcoord[1], y_mcoord[2]-y_mcoord[1]]) # Null5 (DC)
    baseline6 = np.array([x_mcoord[3]-x_mcoord[0], y_mcoord[3]-y_mcoord[0]]) # Null6 (BA)
    baselines = np.array([baseline1, baseline2, baseline3, baseline4, baseline5, baseline6])
    
    if kind == 'ud':
        bl = np.hypot(baselines[:,0], baselines[:,1])
        diameter, wl = args
        vis = createUD(bl, diameter, wl)
        return abs(vis), bl
    elif kind == 'binary':
        diam1, diam2, F1, F2, separation, angular_position, lamb = args
        u, v = baselines[:,0]/lamb, baselines[:,1]/lamb
        vis, phase = createBinary(diam1, diam2, F1, F2, separation, angular_position, u, v, lamb)
        return vis, phase, baselines
    else:
        raise NameError('Unknown kind of object to create. Please choose between \'ud\' and \'binary\'.')
    
def createUD(base, angle, lamb):
    """
    Creates a uniform disk.
    :Parameters:
        **angle: float** 
            angular diameter of the source in mas
        **base: float**
            length of the baseline, in meter
        **lamb: float**
            observing wavelength, in meter
    """
    angle = angle * np.pi / 180. * 0.001 / 3600.
    arg = np.pi * angle * base / (lamb)
    
    return 2 * sp.jv(1, arg) / arg

def createBinary(diam1, diam2, F1, F2, separation, angular_position, u, v, lamb):
    """
    Create a binary system of two uniform disks.
    
    :Parameters:
        **diam1, diam2: floats**
            angular diameters of both components, in mas
        **F1, F2: floats**
            bolometric flux of both components
        **separation: float**
            angular separation between the components, in mas
        **angular_position: float**
            angular position of the component in degree
        **u, v: floats**
            uv-plane coordinates of the baselines
        **lamb: float**
            wavelength in meter
        
    :Returns:
        * a tuple with the absolute values of the visibilities per baselines and the phase (in degree)
    """
    B = np.sqrt(u**2+v**2)
    u /= lamb
    v /= lamb
    disk1 = createUD(B, diam1, lamb)
    disk2 = createUD(B, diam2, lamb)
    angular_position = angular_position * np.pi/180
    x = separation * np.cos(angular_position)
    y = separation * np.sin(angular_position)
    x = x * np.pi / 180. * 0.001 / 3600.
    y = y * np.pi / 180. * 0.001 / 3600.
    binary = (F1 * disk1 + F2*disk2 * np.exp(2.j*(np.pi/lamb)*(u*x + v*y))) / (F1+F2)
    return abs(binary), np.angle(binary, deg=1)
    
def skewedGaussian(x, a, loc, sig, skew):
    gaus = a * np.exp(-(x-loc)**2/(2*sig**2))
    cdf = sp.ndtr(skew * (x-loc)/sig) * a
    return 2 * gaus * cdf
    
if __name__ == '__main__':
    x_mcoord = np.array([2.725, -2.812, -2.469, -0.502]) # x-coordinates of N telescopes in meter
    y_mcoord = np.array([2.317, 1.685, -1.496, -2.363]) # y-coordinates of N telescopes in meter
    baseline1 = np.array([x_mcoord[1]-x_mcoord[0], y_mcoord[1]-y_mcoord[0]]) # Null1 (AD)
    baseline2 = np.array([x_mcoord[2]-x_mcoord[0], y_mcoord[2]-y_mcoord[0]]) # Null2 (AC)
    baseline3 = np.array([x_mcoord[1]-x_mcoord[3], y_mcoord[1]-y_mcoord[3]]) # Null3 (BD)
    baseline4 = np.array([x_mcoord[3]-x_mcoord[2], y_mcoord[3]-y_mcoord[2]]) # Null4 (CB)
    baseline5 = np.array([x_mcoord[2]-x_mcoord[1], y_mcoord[2]-y_mcoord[1]]) # Null5 (DC)
    baseline6 = np.array([x_mcoord[3]-x_mcoord[0], y_mcoord[3]-y_mcoord[0]]) # Null6 (BA)
    baselines = np.array([baseline1, baseline2, baseline3, baseline4, baseline5, baseline6])

    diam1, diam2, F1, F2, separation, angular_position, lamb = 1, 1, 1, 1, 10, 0, 1557e-9
    u, v = baselines[:,0]/lamb, baselines[:,1]/lamb
    vis, phase = createBinary(diam1, diam2, F1, F2, separation, angular_position, u, v, lamb)
    print(abs(vis), phase)
    print(createObject('binary', diam1, diam2, F1, F2, separation, angular_position,lamb))