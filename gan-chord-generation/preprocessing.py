'''
FOR CHORDS

'''
import functools
import numpy as np
import time
import re
import cv2
import torch
import argparse
import os
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from skimage import data, io, filters
import imageio
import time
from torch.autograd import Variable
import cv2 


MUL = 255/12
# C C# D D# E F F# G G# A A# B
# 0 21.25
f = open("chords.txt","r")
line = f.readlines()

line = [i.split() for i in line]

# uni = set(line)
temp_list = functools.reduce(lambda x,y :x+y , line)
# temp_list = map(abs, temp_list) #only want the abs val - elimate -
temp_list = list(set(temp_list))
print(sorted(temp_list))

#(R,G,B) 
# R-(CHORD C-B), 
# G-(MAJOR/MINOR 0,255/2) 
# B(ROOT, SEVENTH, NINTH 0,255/3,2*255/3)

count = 0
map1 = [i*255/12 for i in range(12)]
map2 = ['C', 'C#', 'D', 'D#' ,'E' ,'F', 'F#' ,'G' ,'G#', 'A' ,'A#' ,'B']
# print(map1)

transformed = []
for i in range(len(line)):
    templist = []
    for j in range(len(line[i])):
        # if '#' not in line[i][j] :
        #     print(line[i][j])
        result1 = ''.join([i for i in line[i][j] if not i.isdigit() and i != 'M'])
        result2 = ''.join([i for i in line[i][j] if i.isdigit() ])
        # print(result2)
        # print(result1)
        tempind = map2.index(result1)
        r_val = map1[tempind]
        g_val = 0
        b_val = 0
        if 'M' in line[i][j]:
            g_val = 255/2
        if (result2 == '7'):
            b_val = 255/3
        elif (result2 == '9'):
            b_val = 2*255/3
        temp = [r_val,g_val,b_val]
        # print(temp, line[i][j])
        templist.append(temp)
    transformed.append(templist)

for pic in range(len(transformed)):
    t = np.asarray(transformed[pic])
    print(t.shape)
    sub_dim = int(np.sqrt(t.shape[0]))
    check = sub_dim**2
    t = t[0:check]
    t = t.reshape(sub_dim, sub_dim,3)
    print(type(t), t.shape)
    cv2.imwrite( "./sheetimg/outtest"+str(pic)+".jpg", t )


# imgs = np.expand_dims(t, axis=0)
# imgs = torch.tensor(imgs)
# print(type(imgs), imgs.shape)
# vutils.save_image(imgs, "%s/out_sheet.png", normalize = True)

# def plot_image(tensor):
#     plt.figure()
#     # np arr with the channel dimension -> transpose
#     plt.imshow(tensor.numpy().transpose(1, 2, 0))
#     plt.show()

# plot_image(img)