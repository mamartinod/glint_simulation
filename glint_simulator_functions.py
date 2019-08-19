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

def save(arr, path, date, DIT, na, mag, nbimg, piston, strehl):
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
        f.create_dataset('imagedata', data=arr)

def setZetaCoeff(wl_scale, path, save):
    with h5py.File(path, 'r') as zeta0:
        wl = np.array(zeta0['wl_scale'])
        if np.any(wl>1000):
            wl *= 0.001 # Convert to microns
        
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
        
        zeta_minus = np.array([[np.interp(wl_scale[::-1], wl[::-1], selt[::-1]) for selt in elt] for elt in zeta_minus0])
        zeta_plus = np.array([[np.interp(wl_scale[::-1], wl[::-1], selt[::-1]) for selt in elt] for elt in zeta_plus0])
        
        zeta_minus = zeta_minus[:,:,::-1]
        zeta_plus = zeta_plus[:,:,::-1]
        
    if save:
        with h5py.File('/mnt/96980F95980F72D3/glint/simulation/zeta_coeff2.hdf5', 'w') as newzeta:
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
    x, step = cp.linspace(-25,25, 10000, endpoint=False, retstep=True, dtype=cp.float32)
    cdf = doubleGaussCdf(x, mu1, mu2, sig1, A)
    cdf = cp.asarray(cdf, dtype=cp.float32)
    rv = cp.asnumpy(rv_generator(x, cdf, nsamp))
    rv = np.reshape(rv, shape)
    return rv

if __name__ == '__main__':
    from scipy.optimize import curve_fit
#    x, step = cp.linspace(-100,100, 10000, endpoint=False, retstep=True, dtype=cp.float32)
#    mu1, mu2, sig1, A = 0, 25, 0.5, 1.
#    
#    cdf = doubleGaussCdf(x, mu1, mu2, sig1, A)
#    cdf = cp.asarray(cdf, dtype=cp.float32)
#    rv = cp.asnumpy(rv_generator(x, cdf, 1000000))

    x, step = cp.linspace(-25,25, 10000, endpoint=False, retstep=True, dtype=cp.float32)
    mu1, mu2, sig1, A = 0, 0.8, 0.2, 1.
    rv = rv_gen_doubleGauss((int(1e+6),), mu1, mu2, sig1, A)
    edges = cp.asnumpy(x) - step/2
    edges = np.append(edges, edges[-1]+step)
    
    hist, bin_edges = np.histogram(rv, edges, density=True)
    
    def doubleGaus(x, mu1, mu2, sig1, A):
        out = 1/((2*np.pi)**0.5*(sig1+A*sig1)) * (np.exp(-(x-mu1)**2/(2*sig1**2)) + A*np.exp(-(x-mu2)**2/(2*sig1**2)))
        return out 
    
    x_cpu = cp.asnumpy(x)
    popt, pcov = curve_fit(doubleGaus, x_cpu, hist, [mu1, mu2, sig1, A])
    print(popt)

#    cdf_cpu = cp.asnumpy(cdf)    
#    plt.figure()
#    plt.plot(x_cpu, cdf_cpu)
#    plt.grid()
    
    plt.figure()
    plt.plot(x_cpu, hist)
    plt.plot(x_cpu, cp.asnumpy(doubleGaus(x, *popt)))
    plt.grid()