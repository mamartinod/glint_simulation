
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
        self.isz = isz = 256     # image size (in pixels)
        self.xx, self.yy = np.meshgrid(np.arange(isz)-isz/2, np.arange(isz)-isz/2)
        self.nfr = nfr # number of frames
        
        self.pscale = 5  # image plate scale (in mas/pixel)
        self.wavel1 = wavel1#1.55e-6 # wavelength for image (in meters)
        self.tdiam  = 1.075   # telescope diameter (in meters)
        
        dtor    = np.pi / 180.0 # degree to radian conversion factor
        self.rad2mas = 3.6e6 / dtor  # radian to milliarcsecond conversion factor
        
        ld_r = self.wavel1 / self.tdiam             # lambda/D (in radians)
        ld_p = ld_r * self.rad2mas / self.pscale    # lambda/D (in pixels)
        self.prad = prad = np.round(isz / ld_p / 2.0) # simulated aperture radius (in pixels)
        self.ll     = self.tdiam * isz / (2 * prad) # wavefront extent (in meters)

        
        self.spatial_scale = (np.arange(isz)-isz/2)*self.wavel1/isz*self.rad2mas/self.pscale # in meter, for pupil plane
        self.angular_scale = (np.arange(isz)-isz/2) * self.pscale # in mas, for PSF or image
        
        pupil = np.zeros((isz,isz)) # array of zeros
        
        # Apperture A, D, C, B, resp beam 2, 1, 3, 4
        x_mcoord = [2.725, -2.812, -2.469, -0.502] # x-coordinates of N telescopes in meter
        y_mcoord = [2.317, 1.685, -1.496, -2.363] # y-coordinates of N telescopes in meter
        
        x_pcoord = []
        y_pcoord = []
        for i in range(len(x_mcoord)):
            x0 = x_mcoord[i]/self.wavel1/self.rad2mas * self.pscale*isz
            y0 = y_mcoord[i]/self.wavel1/self.rad2mas * self.pscale*isz
            x_pcoord.append(x0)
            y_pcoord.append(y0)
            pupil[(self.xx-x0)**2 + (self.yy-y0)**2 < prad**2] = 1.0
        
        x_pcoord = np.array(x_pcoord)
        y_pcoord = np.array(y_pcoord)
        
        self.x_pcoord, self.y_pcoord = x_pcoord, y_pcoord

        self.pupil = pupil/np.sum(pupil)
        
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
            return(screen.astype(np.complex64))
            
        L0 = 1e15   # outer-scale (in meters) - very large -> Kolmogorov

        if turbulence_switch:
            self.phase = []
            for k in range(self.nfr):
                phscreen0 = atmo_screen(self.isz, self.ll, r0, L0) # phase-screen for wavelength wavel0
                phscreen1 = phscreen0 * wavel0 / self.wavel1  # phase-screen for wavelength wavel1
                self.phase.append(phscreen1.real)
            self.phase = np.array(self.phase, dtype=np.float32)
        else:
            self.phase = np.zeros((self.nfr, self.isz,self.isz), dtype=np.float32)
                    
    def extractFringePeaks(self):
        phased_pupil = np.zeros_like(self.pupil*np.exp(1j*self.phase))
        for k in range(4):
            x0, y0 = int(np.around(self.x_pcoord[k]+self.isz/2)), int(np.around(self.y_pcoord[k]+self.isz/2))
            y_span, x_span = (y0-int(self.prad), y0+int(self.prad)), (x0-int(self.prad), x0+int(self.prad))
            # We take the average of the phase as beams are injected into single-mode waveguides which only keep the piston mode of the wavefront
            phased_pupil[:,y_span[0]:y_span[1], x_span[0]:x_span[1]] = self.pupil[y_span[0]:y_span[1], x_span[0]:x_span[1]] * np.exp(1j*np.mean(self.phase[:,y_span[0]:y_span[1], x_span[0]:x_span[1]], axis=(1,2))[:,None,None])
        
        fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(phased_pupil, axes=(-2,-1))), axes=(-2,-1)) / np.sqrt(np.prod(self.pupil.shape[1:]))
        fep = np.power(np.abs(fft),2)
        fto = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fep, axes=(-2,-1))), axes=(-2,-1)) * np.sqrt(np.prod(fep.shape[1:]))
 
        self.tf_img = fto * self.tf_obj[None,:,:]

        baseline1 = np.array([self.x_pcoord[1]-self.x_pcoord[0]+self.isz/2, self.y_pcoord[1]-self.y_pcoord[0]+self.isz/2]) # Null1 (AD)
        baseline2 = np.array([self.x_pcoord[2]-self.x_pcoord[0]+self.isz/2, self.y_pcoord[2]-self.y_pcoord[0]+self.isz/2]) # Null2 (AC)
        baseline3 = np.array([self.x_pcoord[1]-self.x_pcoord[3]+self.isz/2, self.y_pcoord[1]-self.y_pcoord[3]+self.isz/2]) # Null3 (BD)
        baseline4 = np.array([self.x_pcoord[3]-self.x_pcoord[2]+self.isz/2, self.y_pcoord[3]-self.y_pcoord[2]+self.isz/2]) # Null4 (CB)
        baseline5 = np.array([self.x_pcoord[2]-self.x_pcoord[1]+self.isz/2, self.y_pcoord[2]-self.y_pcoord[1]+self.isz/2]) # Null5 (DC)
        baseline6 = np.array([self.x_pcoord[3]-self.x_pcoord[0]+self.isz/2, self.y_pcoord[3]-self.y_pcoord[0]+self.isz/2]) # Null6 (BA)
        
        baselines = [baseline1, baseline2, baseline3, baseline4, baseline5, baseline6]
        xx, yy = np.meshgrid(np.arange(self.tf_img.shape[2]), np.arange(self.tf_img.shape[1]))
        sz_peak = self.tdiam/self.wavel1/self.rad2mas * self.pscale * self.isz # Radius of the fringe-peak
        
        diff_phases = []
        for i in range(len(baselines)):
            x0, y0 = baselines[i]
            mask = (xx-x0)**2 + (yy-y0)**2 < sz_peak**2
#            peak0 = self.tf_img[:,mask]
            peak_max = np.argmax(abs(self.tf_img[:,mask]), axis=1)
            peak = np.array([self.tf_img[:,mask][k, peak_max[k]] for k in range(self.tf_img[:,mask].shape[0])])
            diff_phases.append(np.angle(peak))
            
        diff_phases = np.array(diff_phases)
        return diff_phases
        
    def estimateStrehl(self, plot):
        strehl = []
        phased_pupil = self.pupil*np.exp(1j*self.phase)
        window_size = 128
        for k in range(4):
            x0, y0 = int(np.around(self.x_pcoord[k]+self.isz/2)), int(np.around(self.y_pcoord[k]+self.isz/2))
            windowed_pupil = phased_pupil[:,y0-int(self.prad):y0+int(self.prad), x0-int(self.prad):x0+int(self.prad)]
            windowed_pupil = np.pad(windowed_pupil, ((0,0),(0,window_size-int(2*self.prad)),(0,window_size-int(2*self.prad))), mode='constant', constant_values=0.)
            fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(windowed_pupil, axes=(-2,-1))), axes=(-2,-1)) / np.sqrt(np.prod(windowed_pupil.shape[1:]))
            self.fep = fep = np.power(np.abs(fft),2)
            
            windowed_pupil_cali = self.pupil[y0-int(self.prad):y0+int(self.prad), x0-int(self.prad):x0+int(self.prad)]
            windowed_pupil_cali = np.pad(windowed_pupil_cali, ((0,window_size-int(2*self.prad)),(0,window_size-int(2*self.prad))), mode='constant', constant_values=0.)
            fft_cali = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(windowed_pupil_cali, axes=(-2,-1))), axes=(-2,-1)) / np.sqrt(windowed_pupil_cali.size)
            self.fep_cali = fep_cali = np.power(np.abs(fft_cali),2)
            strehl.append(np.max(fep, axis=(-2,-1))/np.max(fep_cali, axis=(-2,-1)))
        strehl = np.array(strehl)
        return strehl
        
    def run(self, r0, wavel_wfs, turb_switch):
        self.createObject(1,1,[0,0], [0,0])
        self.generateAtmosphere(r0, wavel_wfs, turb_switch)
        diff_phases = self.extractFringePeaks()
        strehl = self.estimateStrehl(plot=False)
        return diff_phases, strehl
            
if __name__ == '__main__':
    sz = 1
    wl1 = 1.55e-6
    wl_wfs = 0.5e-6
    r0 = 0.3
    seeing = wl_wfs/r0 * 180/np.pi*3600
    print('Seeing at %.2E nm: %.3f arcsec'%(wl_wfs*1e+9, seeing))
    plop = goas(wl1, 1)
    plop.createObject(sz,sz,[0,0], [0,0])
    print('object created')
    plop.generateAtmosphere(r0, wl_wfs, 1)
    print('atmosphere created')
    phases = plop.extractFringePeaks()
    print('Diff phases got.')
    strehls = plop.estimateStrehl(plot=False)
    print('Strehl got.')
#    plop2 = goas(1.55e-6, 1)
#    plop2.createObject(sz,1,[0,0], [0,0])
#    plop2.generateAtmosphere(0.1, 0.5e-6, 1)
#    plop2.getImage()
#    plop2.extractFringePeaks()
#    plop2.estimateStrehl(plot=False)
#    z2 = plop2.run(0.15, 0.5e-6, 0)    
    
#     histo = np.histogram(phases, int(phases.shape[1]**0.5), density=True)
#     print(phases.std(axis=1)*wl1*1e+6/(2*np.pi))
#     plt.figure();plt.plot(histo[1][:-1], histo[0]);plt.grid()
#     print(strehls.mean(axis=1))
# #    histo = np.histogram(strehls, int(strehls.shape[1]**0.5), density=True)
# #    plt.figure();plt.plot(histo[1][:-1], histo[0]);plt.grid()
    
#     output = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(plop.tf_img, axes=(-2,-1))), axes=(-2,-1))) * np.sqrt(np.prod(plop.tf_obj.shape[1:]))
#     from matplotlib import animation
#     fig = plt.figure()
#     ax = plt.axes()
#     time_text = ax.text(0.05, 0.01, '', transform=ax.transAxes, color='w')
#     im = plt.imshow(output[0],interpolation='none')
#     # initialization function: plot the background of each frame
#     def init():
#         im.set_data(output[0])
#         time_text.set_text('')
#         return [im] + [time_text]
#     # animation function.  This is called sequentially
#     def animate(i):
#         im.set_array(output[i])
#         time_text.set_text('Frame %s/%s'%(i+1, output.shape[0]))
#         return [im] + [time_text]
    
#     anim = animation.FuncAnimation(fig, animate, init_func=init, frames=output.shape[0], interval=500, blit=True) 
