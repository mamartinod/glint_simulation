
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 16:42:53 2019

@author: mamartinod
"""

import numpy as np

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
        
        
