# setting up pytorch
# pip3 install https://download.pytorch.org/whl/cu90/torch-1.0.1-cp37-cp37m-win_amd64.whl #https://pytorch.org/
# https://codeburst.io/quick-guide-for-setting-up-pytorch-in-window-in-2-mins-9342a84704a6

# tutorial #gans
# https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f
# https://medium.com/@rcorbish/sample-gan-using-pytorch-226319052ed1
# https://www.udemy.com/computer-vision-a-z
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# https://towardsdatascience.com/a-beginners-tutorial-on-building-an-ai-image-classifier-using-pytorch-6f85cb69cba7
# https://medium.com/@rcorbish/sample-gan-using-pytorch-226319052ed1
# https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f
# https://www.geeksforgeeks.org/python-denoising-of-colored-images-using-opencv/


'''
2019/5/5
pytorch gans generate new images
'''

from __future__ import print_function
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
import skimage

if __name__ == '__main__':
    torch.multiprocessing.freeze_support() #!!!!!!!!to fix multiprocessing runtime error

    # Root directory for dataset #has to be in another folder??
    dataroot = "pic/"
    filename = dataroot+'1/'+'1.jpg'
    # batch size for training
    batch_size = 16
    # limitation on input image size
    image_size = 64

    # pre-processing the inputs
    transform = transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

    # load the image
    trainset = dset.ImageFolder(dataroot, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
    
    # sanity check
    # num_batches = len(trainloader)
    # print("batch num:",num_batches)
    # display the input ------------------------------------
    # real_batch = next(iter(trainloader))
    # img = real_batch[0][0]
    # print("In folder:",type(img),img.shape)

    # plot the image
    def plot_image(tensor):
        plt.figure()
        # np arr with the channel dimension -> transpose
        plt.imshow(tensor.numpy().transpose(1, 2, 0))
        plt.show()
    
    # plot_image(img)
    # plot_image(torch.randn(inputs.size()[0], 100, 1, 1))

    print(torch.randn(100, 1, 1).shape)
    # plot_image(torch.randn(100, 1, 1))

    # https://www.udemy.com/computer-vision-a-z <-------------------
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.main = nn.Sequential(
                    nn.ConvTranspose2d(4, 512, 4, 1, 0, bias = False), #<-
                    nn.BatchNorm2d(512),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
                    nn.Tanh()
                )
        def forward(self, inputs):
            return self.main(inputs)

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.main = nn.Sequential(
                    nn.Conv2d(3, 64, 4, 2, 1, bias = False),
                    nn.LeakyReLU(0.2, inplace = True),
                    nn.Conv2d(64, 128, 4, 2, 1, bias = False),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2, inplace = True),
                    nn.Conv2d(128, 256, 4, 2, 1, bias = False),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2, inplace = True),
                    nn.Conv2d(256, 512, 4, 2, 1, bias = False),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(0.2, inplace = True),
                    nn.Conv2d(512, 1, 4, 1, 0, bias = False),
                    nn.Sigmoid()
                )      
        def forward(self, inputs):
            return self.main(inputs).view(-1)


    #takes as parameter a neural network and defines its weights
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

            
    print("# init G & D...")
    G = Generator()
    D = Discriminator()
    G.apply(weights_init)
    D.apply(weights_init)

    

    #training the DCGANs
    criterion = nn.BCELoss()
    optimizerG = optim.Adam(G.parameters(), lr = 0.0002, betas = (0.5, 0.999))
    optimizerD = optim.Adam(D.parameters(), lr = 0.0002, betas = (0.5, 0.999))

    epochs = 1000

    if torch.cuda.is_available():
        print("# Cuda...")

    for epoch in range(epochs):
        print("# Epoch [%d/%d]..." % (epoch, epochs))
        for i, data in enumerate(trainloader, 0):            
            #trains the discriminator with a real image
            real, _ = data #change data py list to np tensor

            #test cuda   #Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same   
            #     inputs = Variable(real.cuda()).cuda()
            #     target = Variable(torch.ones(inputs.size()[0]).cuda()).cuda()
            # else:
            #     inputs = Variable(real)
            #     target = Variable(torch.ones(inputs.size()[0]))

            #--------------DDDDDDDDDDDDDDDDDDDDDDDDDDDD--------------------------------------
            # print("#input D:",type(real), real.shape)
            #updates the weights of the discriminator nn
            D.zero_grad()
            # print("...D taking input...", real.shape)
            inputs = Variable(real)
            
            output = D(inputs) #real decision
            # print("...D taking output...", real.shape)
            errD_real = criterion(output, Variable(torch.ones(inputs.size()[0]))) #one = true
            errD_real.backward(retain_graph=True)  #retain graph - for error the buffers have already been freed

            # optimizerD.step() <--

            #train G on D's response
            # G.zero_grad() #update the weights

            noise_input = Variable(torch.randn(inputs.size()[0], 4, 1, 1))
            
 
            # print(noise_input.shape)
            # print(torch.randn(inputs.size()[0], 100, 1, 1).shape)
            # print("--")
            # print(inputs.shape)
            # print(real.shape)

            # input_test = inputs[0].numpy().transpose(1, 2, 0)
            # print(input_test.shape)
            
            d_fake = G(noise_input).detach()
            # d_fake = G(noise_input).detach()
            dg_fake_decision = D(d_fake)
            d_error = criterion(dg_fake_decision, Variable(torch.zeros(inputs.size()[0])))

            #update d
            d_error.backward(retain_graph=True)
            optimizerD.step()

            #--------------GGGGGGGGGGGGGGGGGGGGGGGGGGGG--------------------------------------
            
            #train G based on its true
            G.zero_grad() #update the weights on nn

            g_noise_input = Variable(torch.randn(inputs.size()[0], 4, 1, 1))
            g_fake = G(g_noise_input)
            g_output_decision = D(g_fake)

            g_error = criterion(g_output_decision, Variable(torch.ones(inputs.size()[0]))  )
            
            # back prop
            g_error.backward(retain_graph=True)
            optimizerG.step()

            # print("# noise: ",noise_input.shape)
            # print("# actual: ",inputs.shape)


            # #backpropagating the total error
            # errD = errD_real + errD_fake
            # errD.backward(retain_graph=True) 
            # optimizerD.step()

            # #updates the weights of the generator nn
            # G.zero_grad()

            # G_noise = Variable(torch.randn(inputs.size()[0], 100, 1, 1))
            # target = Variable(torch.ones(inputs.size()[0]))
            # fake = G(G_noise)
            # G_output = D(fake)
            # errG  = criterion(G_output, target)

            # #backpropagating the error
            # errG.backward()
            # optimizerG.step()

            if i % 100 == 0:
                # print(g_fake.data.shape, type(g_fake.data))
                datas = g_fake.data #.detach().numpy()
                transform0 = transforms.ToPILImage()
                after = []
                # print(data.shape, type(data))
                for i in range(len(datas)):
                    data= transform0(datas[i])
                    # data = np.transpose(data, (1, 2, 3, 0)) # put height and width in front
                    # print(data.shape, type(data))
                    # transform = transforms.Compose([transform.resize(10) ])
                    # print(data.shape)
                    transform1 = transforms.Compose([transforms.Scale((10,10))])
                    data = transform1(data)
                    transform2 = transforms.ToTensor()
                    data = transform2(data).detach().numpy()
                    # print(type(data), data.shape)
                    after.append(data)
                
                after = np.asarray(after)
                # print(after.shape, type(after))
                # after = transform2(after)
                data_tensor = torch.from_numpy(after)
                # print(type(data_tensor),data_tensor.shape)


                # vutils.save_image(real, "%s/real_samples.png" % "./results6", normalize = True)
                # vutils.save_image(g_fake.data, "%s/fake_samples_epoch_%03d.png" % ("./results6", epoch), normalize = True)
                if(epoch > 100):
                    for index in range(len(data_tensor)):
                        vutils.save_image(data_tensor[i], "%s/fake_samples_epoch_%03d_%03d.png" % ("./after", epoch,index), normalize = True)

                # print("# Img saved.\n")

    print("# Finished.")



            


    






