# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 09:50:00 2022

@author: thecainfamily
"""



def make_short_otf2(r1,dx,si,ro):
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
    dist=np.sqrt(dist2)
    temp=np.power((dx*dist/ro),(5/3))
    temp3=(np.ones((si,si))-dist/(2*r1/dx)+.0000001)
    binmap=(temp3>0)
    temp4=np.power(np.multiply(temp3,binmap),(1/3))
    temp2=-3.44*np.multiply(temp,temp4)
    otf=np.exp(temp2)
    otf2=fftshift(np.multiply(otf,binmap))
    return(otf2)
short_otf=make_short_otf2(5,20/3000,3000,2)


