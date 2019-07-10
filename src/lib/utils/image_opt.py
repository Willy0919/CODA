#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Willy
import matplotlib.pyplot as plt
import torch
import numpy as np
from skimage import transform as sk_transform
from skimage import io, color
import os,cv2
import scipy.io as scio

def show(image,dots,sigma):
    _dots = dots.astype(np.int)
    plt.imshow(image)
    plt.scatter(_dots[:, 0], _dots[:, 1], s=10, marker='.', c='r')
    plt.show()

def getPerspective(dots, pmap):
    persp = []
    for i in range(len(dots)):
        x,y = dots[i].astype(np.int32)
        #g = 1/pmap[x, y]
        persp.append(pmap[y, x]*0.185)

    return torch.FloatTensor(np.array(persp))

def findSigma(dots,k,beta):
    PN2 = torch.FloatTensor(dots)
    AB = torch.mm(PN2,torch.t(PN2))
    AA = torch.unsqueeze(torch.diag(AB),1)
    DIST = torch.sqrt(AA - 2*AB+AA.t())
    sorted,indices = torch.sort(DIST)
    sigma = beta * torch.mean(sorted[:,1:1+k],1)
    return sigma

def fspecial(shape=(3,3),sigma=0.5):
    m, n = (shape[0] - 1.) / 2., (shape[1] -1)/2.
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    #h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
       h /= sumh
    return h


def genDensity(image, dots, sigmas, margin_size,rescale = 0.125):
    '''
    1) the real num < generated num
    partial boundary energy loss
    2) accelerate
    '''
    h, w = image.shape[:2]

    dmap_extend = np.zeros((h + margin_size - 1, w + margin_size - 1), np.float32)
   
    margin = int((margin_size - 1) / 2)
   
    for i in range(len(sigmas)):
        cx,cy = dots[i].astype(np.int)
        sigma = sigmas[i]
        kernel_size = int(min(1*sigma,margin))
        gaussian_kernel = fspecial((kernel_size, kernel_size), sigma)
        
        dmap_extend[cy+margin:cy+margin + kernel_size, cx+margin:cx+margin + kernel_size] += gaussian_kernel
    dmap = dmap_extend[margin:margin+h, margin:margin+w]
    
    if  rescale!=1.0:
        dmap = sk_transform.resize(dmap,(int(h*rescale),int(w*rescale)),preserve_range=True)
        dmap/= (rescale**2)
    return dmap

def getAttentionDensity(image,nlevel, dots,  sigmas, margin_size,rescale = 0.125):
    h, w = image.shape[:2]
    dmap = []
    levels = [3,9,27]
    for i in range(nlevel):
        level = levels[i]

        dmap_extend = np.zeros((h + margin_size - 1, w + margin_size - 1), np.float32)
        margin = int((margin_size - 1) / 2)
        for j in range(len(dots)):
            cx, cy = dots[j].astype(np.int)
            att = sigmas[j][i]
            kernel_size = int(min(level, margin))
            gaussian_kernel = fspecial((kernel_size, kernel_size), level)
            dmap_extend[cy + margin:cy + margin + kernel_size, cx + margin:cx + margin + kernel_size] += gaussian_kernel*att
        sub_dmap =  dmap_extend[margin:margin+h, margin:margin+w]
        if rescale != 1.0:
            sub_dmap = sk_transform.resize(dmap, (int(h * rescale), int(w * rescale)), preserve_range=True)
            sub_dmap /= (rescale ** 2)

        dmap.append(sub_dmap)

    return np.array(dmap)

def getMaskedDots(image, dots, mask):
    
    dot_map = np.zeros(image.shape[:2], dtype=np.uint32)
    dot_map[dots[:,1].astype(np.int32), dots[:,0].astype(np.int32)] = 1

    dot_map = dot_map*mask
    dot_mask = []

    idx = np.where(dot_map == 1)
  
    for i in range(len(idx[0])):
        dot_mask.append([idx[1][i], idx[0][i]])

    return np.array(dot_mask)


def showTest(image, pre_dens, gt_dens):
    for i in range(image.shape[0]):
        img = torch.squeeze(image[i,:], 0)
        img = (img * 127.5 + 127.5).numpy().astype(np.uint8)
        img = img.transpose((1, 2, 0))
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.set_title('original image', fontsize=14)
        ax.imshow(img)

        ax = fig.add_subplot(223)
        gt_densmap = torch.squeeze(gt_dens[i, :], 0)
        ax.set_title('gt count: {:.2f}'.format(np.sum(gt_densmap.numpy())))
        ax.imshow(gt_densmap.numpy(), cmap='jet', interpolation='bilinear')
        ax = fig.add_subplot(224)
        pre_densmap = torch.squeeze(pre_dens[i,:],0)
        ax.set_title('predict count: {:.2f}'.format(np.sum(pre_densmap.numpy())))
        ax.imshow(pre_densmap.numpy(), cmap='jet', interpolation='bilinear')
        plt.show()
        print('gt = ',np.sum(gt_densmap.numpy()),'pre = ',np.sum(pre_densmap.numpy()))


def showGt(sample):
    
    images = sample['image']
    for i in range(images.shape[0]):
        image = torch.squeeze(sample['image'][i, :], 0)
        image = (image * 127.5 + 127.5).numpy().astype(np.uint8)
        _dots = sample['dots'][i, :].numpy().astype(np.int)
        image = image.transpose((1, 2, 0))
        print('images shape:', image.shape[0]*image.shape[1])
        
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.set_title('original image', fontsize=14)
        ax.imshow(image)
        count = sample['count'][i]
        ax.set_title('gt count: {}'.format(count))
        if count > 0:
            
            ax.scatter(_dots[:count, 0], _dots[:count, 1], s=10, marker='.', c='r')
        ax = fig.add_subplot(122)
        densityMap = torch.squeeze(sample['densityMap'][i, :], 0)
        ax.set_title('densityMap : {:.2f}'.format(np.sum(densityMap.numpy())))
        ax.imshow(densityMap.numpy(), cmap='jet', interpolation='bilinear')        
        plt.show()

def showMultiscale(sample):
   
    images = sample['image']
    for i in range(images.shape[0]):
        image = torch.squeeze(sample['image'][i, :], 0)
        image = (image * 127.5 + 127.5).numpy().astype(np.uint8)
        _dots = sample['dots'][i, :].numpy().astype(np.int)
        image = image.transpose((1, 2, 0))

        multi_img = torch.squeeze(sample['scale_images'][i, :], 0)
    
        densityMap = torch.squeeze(sample['densityMap'][i, :], 0)
        dmap_sum = densityMap#[0] + densityMap[1] + densityMap[2]# + densityMap[3]
        
        fig = plt.figure()
        ax = fig.add_subplot(331)
        ax.set_title('original image', fontsize=14)
        ax.imshow(image)
        count = sample['count'][i]
        ax.set_title('gt count: {}'.format(count))
        if count > 0:
            
            ax.scatter(_dots[:count, 0], _dots[:count, 1], s=10, marker='.', c='r')

        ax = fig.add_subplot(332)

        ax.set_title('densityMap : {:.2f}'.format(np.sum(dmap_sum.numpy())))
        ax.imshow(dmap_sum.numpy(), cmap='jet', interpolation='bilinear')
        for j in range(multi_img.shape[0]):
            ax = fig.add_subplot(333+j)

            scale_img = (multi_img[j] * 127.5 + 127.5).numpy().astype(np.uint8)
            scale_img = scale_img.transpose((1, 2, 0))
            ax.set_title('scaleimg{} : '.format(j))
            ax.imshow(scale_img)


        plt.show()

