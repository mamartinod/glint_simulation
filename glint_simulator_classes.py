
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 16:42:53 2019

@author: mamartinod
"""

import numpy as np
import matplotlib.pyplot as plt

class Noising:
    def __init__(self, nbimg, inpt='mock', xsize=None, ysize=None):
        if inpt != 'mock':
            self.image = inpt
        else:
            if xsize != None and ysize!=None:
                self.image = np.zeros((nbimg,ysize,xsize))
            else:
                raise('Size of the detector cannot be None')
            
        self.bruitph = None
        self.ron = None
        self.converted = None
        self.output = None
        self.nbimg = nbimg
    
        
    def __addPoissonNoise(self, Ndark, active):
        if active: 
            dark = Ndark * np.ones(self.image.shape)
            self.bruitph = np.random.poisson(dark+self.image)
        else:
            self.bruitph = self.image
        
    def __addRON(self, gainsys, offset, sigma_ron, active):
        if active:
            shape = self.image.shape
            signal = self.bruitph * gainsys # ADU
            ron = np.random.normal(0, sigma_ron, (self.nbimg, shape[1], shape[2]))
            signal = signal + offset + ron
            self.ron = signal
        else:
            self.ron = self.bruitph
        
    def __convert16bits(self, convert):
        converted = self.ron.copy()
        if convert:
            self.converted = converted.astype(np.uint16)
            if np.any(self.converted>2**16):
                print('WARNING: saturation of the detector')
        else:
            self.converted = converted
        
    def addNoise(self, Ndark, gainsys, offset, sigma_ron, active=[True, True], convert=True):
        self.__addPoissonNoise(Ndark, active[0])
        self.__addRON(gainsys, offset, sigma_ron, active[1])
        self.__convert16bits(convert)
        self.output = self.converted
        
        
class goas(object):
    """
    GLINT Object and Atmosphere Simulator
    """
    def __init__(self, wavel1, nfr):
        """
        Initialize parameters and properties of the aperture mask
        """
        self.isz = isz = 80     # image size (in pixels)
        self.xx, self.yy = np.meshgrid(np.arange(isz)-isz/2, np.arange(isz)-isz/2)
        self.nfr = nfr # number of frames
        
        self.pscale = 65  # image plate scale (in mas/pixel)
        self.wavel1 = wavel1#1.55e-6 # wavelength for image (in meters)
        self.tdiam  = 1.075   # telescope diameter (in meters)
        
        dtor    = np.pi / 180.0 # degree to radian conversion factor
        self.rad2mas = 3.6e6 / dtor  # radian to milliarcsecond conversion factor
        
        ld_r = self.wavel1 / self.tdiam             # lambda/D (in radians)
        ld_p = ld_r * self.rad2mas / self.pscale    # lambda/D (in pixels)
        prad = np.round(isz / ld_p / 2.0) # simulated aperture radius (in pixels)
        self.ll     = self.tdiam * isz / (2 * prad) # wavefront extent (in meters)

        
        self.spatial_scale = (np.arange(isz)-isz/2)*self.wavel1/isz*self.rad2mas/self.pscale # in meter, for pupil plane
        self.angular_scale = (np.arange(isz)-isz/2) * self.pscale # in mas, for PSF or image
        
        pupil = np.zeros((isz,isz)) # array of zeros
        
        # Apperture A, D, C, B, resp beam 2, 1, 3, 4
        x_mcoord = [2.725, -2.812, -2.469, -0.502] # x-coordinates of N telescopes in meter
        y_mcoord = [2.317, 1.685, -1.496, -2.363] # y-coordinates of N telescopes in meter
        
        x_pcoord = []
        y_pcoord = []
        self.mono_pupils = []
        for i in range(len(x_mcoord)):
            x0 = x_mcoord[i]/self.wavel1/self.rad2mas * self.pscale*isz
            y0 = y_mcoord[i]/self.wavel1/self.rad2mas * self.pscale*isz
            x_pcoord.append(x0)
            y_pcoord.append(y0)
            pupil[(self.xx-x0)**2 + (self.yy-y0)**2 < prad**2] = 1.0
            mono_pupil = np.zeros(pupil.shape)
            mono_pupil[(self.xx-x0)**2 + (self.yy-y0)**2 < prad**2] = 1.0
            self.mono_pupils.append(mono_pupil)
        
        x_pcoord = np.array(x_pcoord)
        y_pcoord = np.array(y_pcoord)
        self.mono_pupils = np.array(self.mono_pupils)
        
        self.x_pcoord, self.y_pcoord = x_pcoord, y_pcoord

        self.pupil = pupil/np.sum(pupil)
        
#        self.pupil = np.tile(self.pupil[None,:,:], (self.nfr, 1, 1))
#        self.mono_pupils = np.tile(self.mono_pupils[:,None,:,:], (1,self.nfr,1,1))

    def createObject(self, objdiam1, objdiam2, coord1, coord2):
        objrad1 = objdiam1 / 2. / self.pscale # Radius in pixel
        objrad2 = objdiam2 / 2. / self.pscale # Radius in pixel
        
        xp1, yp1 = coord1
        xp2, yp2 = coord2
        
        xp1, yp1 = xp1/self.pscale, yp1/self.pscale
        xp2, yp2 = xp2/self.pscale, yp2/self.pscale
        
        obj = np.zeros((self.isz, self.isz))
        obj[(self.xx-xp1)**2/objrad1**2 + (self.yy-yp1)**2/objrad2**2 <= 1.] = 1.
        obj = obj/np.sum(obj)
        self.obj = obj
        
        self.tf_obj = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(obj, axes=(-2,-1))), axes=(-2,-1)) / np.sqrt(np.prod(obj.shape[1:]))
        
    def generateAtmosphere(self, r0, wavel0, turbulence_switch):
        """
        Generate an atmospheric turbulence according to a Kolmogorov model
        
        :Parameters:
            * **r0**: float, Fried's parameter in meter.
            * **turbulence_switch**: bool, activate the turbulence or not.
        """
        def atmo_screen(isz, ll, r0, L0):
            '''The Kolmogorov - Von Karman phase screen generation algorithm.
        
            Adapted from the work of Carbillet & Riccardi (2010).
            http://cdsads.u-strasbg.fr/abs/2010ApOpt..49G..47C
        
            Parameters:
            ----------
        
            - isz: the size of the array to be computed (in pixels)
            - ll:  the physical extent of the phase screen (in meters)
            - r0: the Fried parameter, measured at a given wavelength (in meters)
            - L0: the outer scale parameter (in meters)
        
            Returns: two independent phase screens, available in the real and imaginary
            part of the returned array.
        
            -----------------------------------------------------------------
            Credit : Frantz Martinache
            '''
            phs = 2*np.pi * (np.random.rand(isz, isz) - 0.5)
        
            xx, yy = np.meshgrid(np.arange(isz)-isz/2, np.arange(isz)-isz/2)
            rr = np.hypot(yy, xx)
            rr = np.fft.fftshift(rr)
            rr[0,0] = 1.0
        
            modul = (rr**2 + (ll/L0)**2)**(-11/12.)
            screen = np.fft.ifft2(modul * np.exp(1j*phs)) * isz**2
            screen *= np.sqrt(2*0.0228)*(ll/r0)**(5/6.)
        
            screen -= screen.mean()
            return(screen)
            
#        wavel0 = 1.55e-6 # wavelength where r0 is measured (in meters)
        L0     = 1e15   # outer-scale (in meters) - very large -> Kolmogorov

        if turbulence_switch:
            self.phase = []
            for k in range(self.nfr):
                phscreen0 = atmo_screen(self.isz, self.ll, r0, L0) # phase-screen for wavelength wavel0
                phscreen1 = phscreen0 * wavel0 / self.wavel1  # phase-screen for wavelength wavel1
                self.phase.append(phscreen1.real)
            self.phase = np.array(self.phase)
        else:
            self.phase = np.zeros((self.nfr, self.isz,self.isz))
            
    def getImage(self):
        fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(self.pupil*np.exp(1j*self.phase), axes=(-2,-1))), axes=(-2,-1)) / np.sqrt(np.prod(self.pupil.shape[1:]))
        fep = np.power(np.abs(fft),2)
        fft, self.fep = fft, fep
        fto = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fep, axes=(-2,-1))), axes=(-2,-1)) * np.sqrt(np.prod(fep.shape[1:]))

        fft_calib = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(self.pupil, axes=(-2,-1))), axes=(-2,-1)) / np.sqrt(np.prod(self.pupil.shape[1:]))
        fep_calib = np.power(np.abs(fft_calib),2)
        self.fto_calib = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fep_calib, axes=(-2,-1))), axes=(-2,-1)) * np.sqrt(np.prod(fep_calib.shape[1:]))
        
        self.tf_img = fto * self.tf_obj[None,:,:]
#        self.img = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(self.tf_img, axes=(-2,-1))), axes=(-2,-1))) * np.sqrt(np.prod(self.tf_obj.shape[1:]))
        
    def extractFringePeaks(self):
        baseline1 = np.array([self.x_pcoord[1]-self.x_pcoord[0]+self.isz/2, self.y_pcoord[1]-self.y_pcoord[0]+self.isz/2]) # Null1 (AD)
        baseline2 = np.array([self.x_pcoord[2]-self.x_pcoord[0]+self.isz/2, self.y_pcoord[2]-self.y_pcoord[0]+self.isz/2]) # Null2 (AC)
        baseline3 = np.array([self.x_pcoord[1]-self.x_pcoord[3]+self.isz/2, self.y_pcoord[1]-self.y_pcoord[3]+self.isz/2]) # Null3 (BD)
        baseline4 = np.array([self.x_pcoord[3]-self.x_pcoord[2]+self.isz/2, self.y_pcoord[3]-self.y_pcoord[2]+self.isz/2]) # Null4 (CB)
        baseline5 = np.array([self.x_pcoord[2]-self.x_pcoord[1]+self.isz/2, self.y_pcoord[2]-self.y_pcoord[1]+self.isz/2]) # Null5 (DC)
        baseline6 = np.array([self.x_pcoord[3]-self.x_pcoord[0]+self.isz/2, self.y_pcoord[3]-self.y_pcoord[0]+self.isz/2]) # Null6 (BA)
        
        self.baselines = [baseline1, baseline2, baseline3, baseline4, baseline5, baseline6]
        xx, yy = np.meshgrid(np.arange(self.tf_img.shape[2]), np.arange(self.tf_img.shape[1]))
        self.sz_peak = self.tdiam/self.wavel1/self.rad2mas * self.pscale * self.isz # Radius of the fringe-peak
        
        centre = np.unravel_index(np.argmax(abs(self.tf_img)), self.tf_img.shape)
        diff_phases = []
        visibilities = []
        for i in range(len(self.baselines)):
            x0, y0 = self.baselines[i]
            mask = (xx-x0)**2 + (yy-y0)**2 < self.sz_peak**2
            mask_central = (xx-centre[1])**2 + (yy-centre[0])**2 < self.sz_peak**2
            mask = np.tile(mask, (self.nfr, 1, 1))
            mask_central = np.tile(mask_central, (self.nfr, 1, 1))
#            self.mask = mask            
#            self.mask_central = mask_central
            
            mask_tf_img = self.tf_img.copy()
            maskcentral_tf_img = self.tf_img.copy()            
            mask_tf_img[~mask] = 0
            maskcentral_tf_img[~mask_central] = 0
                
            mask_calib = self.fto_calib.copy()
            maskcentral_calib = self.fto_calib.copy()
            mask_calib[~mask[0]] = 0
            maskcentral_calib[~mask_central[0]] = 0    
            
            chunk = np.sum(abs(mask_tf_img), axis=(-2,-1)) / np.max(abs(maskcentral_tf_img), axis=(-2,-1))
            chunk_calib = np.sum(abs(mask_calib)) / np.max(abs(maskcentral_calib))
            
            peak = np.sum(mask_tf_img, axis=(-2,-1))
            diff_phases.append(np.angle(peak))
            visibilities.append(chunk/chunk_calib)            
            
        diff_phases = np.array(diff_phases)
        visibilities = np.array(visibilities)
        self.diff_phases = diff_phases
        self.visibilities = visibilities
        
    def estimateStrehl(self, plot):
        self.strehl = []
        for k in range(self.mono_pupils.shape[0]):
            fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(self.mono_pupils[k]*np.exp(1j*self.phase), axes=(-2,-1))), axes=(-2,-1)) / np.sqrt(np.prod(self.mono_pupils[k].shape[1:]))
            fep = np.power(np.abs(fft),2)        
            fft_cali = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(self.mono_pupils[k], axes=(-2,-1))), axes=(-2,-1)) / np.sqrt(np.prod(self.mono_pupils[k].shape[1:]))
            fep_cali = np.power(np.abs(fft_cali),2)
            self.strehl.append(np.max(fep, axis=(-2,-1))/np.max(fep_cali, axis=(-2,-1)))
            if plot:
                plt.figure()
                plt.subplot(121)
                plt.imshow(fep**0.25)
                plt.subplot(122)
                plt.imshow(fep_cali**0.25)
        self.strehl = np.array(self.strehl)
        
    def run(self, r0, wavel_wfs, turb_switch):
        self.createObject(1,1,[0,0], [0,0])
        self.generateAtmosphere(r0, wavel_wfs, turb_switch)
        self.getImage()
        self.extractFringePeaks()
        self.estimateStrehl(plot=False)
        return self.diff_phases, self.strehl
            
if __name__ == '__main__':
    sz = 150
    plop = goas(4.3e-6, 1)
    plop.createObject(sz,sz,[0,0], [0,0])
    plop.generateAtmosphere(0.15, 0.5e-6, 0)
    plop.getImage()
    plop.extractFringePeaks()
    plop.estimateStrehl(plot=False)
#    z = plop.run(0.15, 0.5e-6, 0)
    
#    plop2 = goas(1.55e-6, 1)
#    plop2.createObject(sz,1,[0,0], [0,0])
#    plop2.generateAtmosphere(0.1, 0.5e-6, 1)
#    plop2.getImage()
#    plop2.extractFringePeaks()
#    plop2.estimateStrehl(plot=False)
#    z2 = plop2.run(0.15, 0.5e-6, 0)    