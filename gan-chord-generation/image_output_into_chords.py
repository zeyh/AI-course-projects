'''

'''
import numpy as np 
import cv2 
from matplotlib import pyplot as plt 
import torchvision.utils as vutils
import torch
import functools
import scipy
from scipy.stats import entropy
import math


imgname = "fake_samples_epoch_"
filetype = ".png"
rootdir = "./after/"
imgnum1 = '999' + "_"
imgnum2 = '008'
img = cv2.imread(rootdir+imgname+imgnum1+imgnum2+filetype)
b,g,r = cv2.split(img)           # get b,g,r
rgb_img = cv2.merge([r,g,b])  


rgb_img_len = rgb_img.shape[0]
rgb_img = rgb_img.reshape(rgb_img_len*rgb_img_len, 3)
print(rgb_img.shape, type(rgb_img))
# print(rgb_img)

map1 = [i*255/12 for i in range(12)]
map2 = ['C', 'C#', 'D', 'D#' ,'E' ,'F', 'F#' ,'G' ,'G#', 'A' ,'A#' ,'B']

# print(map1)
#(R,G,B) 
# R-(CHORD C-B), 
# G-(MAJOR/MINOR 0,255/2) 
# B(ROOT, SEVENTH, NINTH 0,255/3,2*255/3)

out = []

for i in range(len(rgb_img)):
    strtemp = []
    key = rgb_img[i][0] / (255/12)
    strtemp.append(map2[int(key)])
    if rgb_img[i][1] < 225/2:
        strtemp.append("m")
    if rgb_img[i][2] > 225/2 and rgb_img[i][2] < 2*225/2:
        strtemp.append("7")
    elif rgb_img[i][2] > 2*225/2:
        strtemp.append("9")
    strfin = "".join(strtemp) 
    out.append(strfin)
print(out)
print("----")
f = open("chords.txt","r")
line = f.readlines()

line = [i.split() for i in line]
# uni = set(line)
temp_list = functools.reduce(lambda x,y :x+y , line)
# temp_list = map(abs, temp_list) #only want the abs val - elimate -
temp_list = list(set(temp_list))
# print(sorted(temp_list))

filteredout = []
for i in out:
    rannum = np.random.uniform()
    if i not in temp_list and rannum < 0.8:
        pass
    else:
        filteredout.append(i)
print(len(filteredout),filteredout)


#https://stackoverflow.com/questions/2979174/how-do-i-compute-the-approximate-entropy-of-a-bit-string
def entropy1(string):
        "Calculates the Shannon entropy of a string"

        # get probability of chars in string
        prob = [ float(string.count(c)) / len(string) for c in dict.fromkeys(list(string)) ]

        # calculate the entropy
        entropy = - sum([ p * math.log(p) / math.log(2.0) for p in prob ])

        return entropy


#---------------------------------
print("----")
# print(scipy.stats.entropy(out))
# tempstr = functools.reduce(lambda x,y :x+y , out)
# tempstr = functools.reduce(lambda x,y :x+y , filteredout)
# tempstr = functools.reduce(lambda x,y :x+y , temp_list)
lenlist = [len(filteredout),len(out),len(temp_list)]

shortest = min(lenlist)
print("filtered out: ",entropy1(filteredout[:shortest]))
print("raw out: ",entropy1(out[:shortest]))
print("input: ",entropy1(temp_list[:shortest]))

# print(len(temp_list[:shortest]))
# print(len(out[:shortest]))
# print(len(filteredout[:shortest]))

#https://www.apronus.com/music/onlineguitar.htm
#hear the chord..... ^