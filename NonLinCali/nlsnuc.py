import numpy as np
from PIL import Image, ImageSequence
import numpy as np
import os
file_path = "NonLinCali\inolight2.tif"
data0=np.zeros((100,1000,1000))
img1=Image.open(file_path)
for ii in range(0,100):
    img1.seek(ii)
    temp_img=np.array(img1)
    data0[ii]=temp_img
file_path = ("ilight40.tif")
data40=np.zeros((100,1000,1000))
img1=Image.open(file_path)
for ii in range(0,100):
    img1.seek(ii)
    temp_img=np.array(img1)
    data40[ii]=temp_img
file_path = ("ilight100.tif")
data100=np.zeros((100,1000,1000))
img1=Image.open(file_path)
for ii in range(0,100):
    img1.seek(ii)
    temp_img=np.array(img1)
    data100[ii]=temp_img
file_path = ("ilight1.tif")
data1=np.zeros((100,1000,1000))
img1=Image.open(file_path)
for ii in range(0,100):
    img1.seek(ii)
    temp_img=np.array(img1)
    data1[ii]=temp_img
file_path = ("ilight2.tif")
data2=np.zeros((100,1000,1000))
img1=Image.open(file_path)
for ii in range(0,100):
    img1.seek(ii)
    temp_img=np.array(img1)
    data2[ii]=temp_img
D100=np.median(data100-52,0)
D40=np.median(data40-52,0)
ratio=np.divide(D100,D40)
alpha_est=np.zeros((1000,1000))
ii=100
jj=100
dataA=data40[:,ii,jj]
dataB=data100[:,ii,jj]
m1=np.mean(dataA,0)
m2=np.mean(dataB,0)
v1=np.var(dataA,0)
v2=np.var(dataB,0)
Gmat=np.divide((v2-v1),(m2-m1))
G=np.median(Gmat)
Kmat=(m2-m1)/(60*G)
Bmat=m1-40*Kmat*G