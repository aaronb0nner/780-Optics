# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 11:22:08 2025


"""

def make_long_otf(D,dx,si,ro):
    import numpy as np
    from scipy.fft import fftshift
    pupilx=np.zeros((si,si))
    otf=np.zeros((si,si))
    if(2*np.floor(si/2)==si):
        mi=int(np.floor(si/2))
        pupilx=np.zeros([si,si])
        
        for i in range(0,si-1):
            pupilx[i]=range(-mi,mi)
           
    if(2*np.floor(si/2)!=si):
         mi=int(np.floor(si/2))
         pupilx=np.zeros([si,si])
         for i in range(0,si-1):
             pupilx[i]=range(-mi,mi+1)
    pupily=np.transpose(pupilx)
    dist2=np.multiply(pupilx,pupilx)+np.multiply(pupily,pupily)
    dist=dx*np.sqrt(dist2)
    binmap=(dist<D)
    temp=np.power((dist/ro),5/3)
    temp2=-3.44*temp
    otf=np.multiply(np.exp(temp2),binmap)
    otf2=fftshift(otf)
    return(otf2)
