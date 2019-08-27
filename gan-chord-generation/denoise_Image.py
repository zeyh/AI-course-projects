import numpy as np 
import cv2 
from matplotlib import pyplot as plt 
import torchvision.utils as vutils
import torch

#https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Non-local_Means_Denoising_Algorithm_Noise_Reduction.php
imgnum = '11'
img = cv2.imread(imgnum+'.png')
b,g,r = cv2.split(img)           # get b,g,r
rgb_img = cv2.merge([r,g,b])     # switch it to rgb

# Denoising
dst = cv2.fastNlMeansDenoisingColored(img,None,6,10,10,8)
b,g,r = cv2.split(dst)           # get b,g,r
rgb_dst = cv2.merge([r,g,b])     # switch it to rgb

# print(type(rgb_dst), rgb_dst.shape)
# nrgb_dst = np.transpose(rgb_dst,(2,0,1))
# after = torch.from_numpy(rgb_dst)
# # after, _ = rgb_dst 
# print(type(after))
# vutils.save_image(after, "%s/denoised.png" % ".", normalize = True)

# plt.imshow(rgb_dst)
# plt.show()

#flip the rgb again...
b,g,r = cv2.split(rgb_dst)           # get b,g,r
rgb_dst = cv2.merge([r,g,b])     # switch it to rgb
print(rgb_dst.shape, type(rgb_dst))
cv2.imwrite( "./denoised/denoised"+imgnum+".jpg", rgb_dst )
