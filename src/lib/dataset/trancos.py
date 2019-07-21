#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Willy
from __future__ import print_function, division
from skimage import io, color
from skimage import transform as sk_transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import torch
import matplotlib.pyplot as plt
import scipy.io as scio
import warnings
import numpy as np
warnings.filterwarnings("ignore")
import random
import h5py
from src.lib.utils.image_opt import findSigma, fspecial,showMultiscale, genDensity, showSample,showLevelGt,getMaskedDots,showGt,getPerspective

class IsColor_t(object):
    def __init__(self,color=True):
        self.color = color
    def __call__(self,sample):
        image = sample['image']
        if len(image.shape)==2:
            if self.color:
                image = color.gray2rgb(image)
        else:
            if not self.color:
                image = color.rgb2gray(image)
                _image = np.zeros((image.shape[0],image.shape[1],3),np.uint8)
                _image[:, :, 0] = image*255
                _image[:, :, 1] = image*255
                _image[:, :, 2] = image*255
                image = _image
        sample['image'] = image
        return sample


class RandomFlip_t(object):
    def __call__(self, sample):
        if random.random() < 0.5:
            image = sample['image']
            sample['image'] = sample['image'][:,::-1]
            if len(sample['dots'])!= 0:
                sample['dots'][:,0] = image.shape[1] -1 - sample['dots'][:,0]
        return sample

class PreferredSize_t(object):
    def __init__(self,size = 0,use_multiscale=False):
        self.size = size
        self.use_multiscale=use_multiscale

    def __call__(self,sample):
        if self.size>0:
            image = sample['image']
            mask = sample['mask']
            h,w,c = image.shape
            ratio = 1

            if h > w:
                new_h, new_w = self.size, int(self.size * w / h+0.5)
                ratio = self.size/h
            else:
                new_h, new_w = int(self.size * h / w+0.5), self.size
                ratio = self.size/w

            if self.use_multiscale:
                multi_img = sample['scale_images']
                multi_img_new = []
                for img_s in multi_img:
                    h_s, w_s, c_s = img_s.shape
                    if h_s > w_s:
                        new_h_s, new_w_s = self.size, int(self.size * w_s / h_s + 0.5)
                        ratio_s = self.size / h_s
                    else:
                        new_h_s, new_w_s = int(self.size * h_s / w_s + 0.5), self.size
                        ratio_s = self.size / w_s

                    resized_img_s = sk_transform.resize(img_s, (new_h_s, new_w_s), preserve_range=True)
                    out_img_s = np.zeros((self.size, self.size, c_s), dtype=np.float32)
                    out_img_s[...] = 127.5
                    out_img_s[:new_h_s, :new_w_s, :] = resized_img_s
                    multi_img_new.append(out_img_s)

                sample['scale_images'] = np.array(multi_img_new)

            sample['dots'] = sample['dots']*ratio
            resized_image = sk_transform.resize(image,(new_h,new_w),preserve_range = True)
            resized_mask = sk_transform.resize(mask,(new_h,new_w),preserve_range = True)
            out_image = np.zeros((self.size,self.size,c),dtype=np.float32)
            out_image[...] = 127.5
            out_image[:new_h,:new_w,:] = resized_image

            out_mask = np.zeros((self.size,self.size),dtype=np.float32)
            out_mask[...] = 0
            out_mask[:new_h, :new_w] = resized_mask
            sample['image'] = out_image
            sample['mask'] = out_mask
        return sample


class Multiscale_t(object):
    def __init__(self,cropscale=[]):
        self.cropscale = cropscale


    def __call__(self,sample):
        image = sample['image']

        h,w = image.shape[:2]
        cx = int(w/2)
        cy = int(h/2)
        scale_img = []
        for i in self.cropscale:
            scale_img.append(image[cy - int(h*i/2): cy + int(h*i/2),
                cx - int(w*i/2): cx + int(w*i/2)])

        sample['scale_images'] = np.array(scale_img)

        return sample

class NineCrop_t(object):
    def __call__(self,sample):
        image = sample['image']
        dots = sample['dots']
        sigmas = sample['sigma']
        h,w = image.shape[:2]
        i = random.randint(0,2)
        j = random.randint(0,2)
        left = int(w/4*i)
        top = int(h/4*j)
        width = int(w/2)
        height = int(h/2)

        image = image[top: top + height,
                left: left + width]
        if len(dots)!=0:
            idx = np.where(
                (dots[:, 0] >= left) & (dots[:, 1] >= top) & (dots[:, 0] < left+width) & (dots[:, 1] < top+height))
            dots = dots[idx]
            dots[:,0] -= left
            dots[:,1] -= top
            idx = idx[0].tolist()
            if len(idx)==0:
                sigmas = torch.FloatTensor([])
            else:
                sigmas = sigmas.index_select(0,torch.LongTensor(idx))

        sample['image'] = image
        sample['dots'] = dots
        sample['sigma'] = sigmas
        return sample

class ToTensor_t(object):
    def __init__(self,rescale=1.0,margin_size = 1001,max_dot = 4000,use_perspective=True,use_multiscale=False):
        self.rescale = rescale
        self.margin_size = margin_size
        self.max_dot = max_dot
        self.use_perspective = use_perspective
        self.use_multiscale = use_multiscale

    def __call__(self,sample):
        image = sample['image']

        dots = sample['dots']
        sigmas = sample['sigma']
        if len(dots) == 0:
            densityMap = np.zeros(image.shape[:2]).astype(np.float32)
        else:
            densityMap = genDensity(image, dots, sigmas, self.margin_size, rescale=1.0)
            densityMap = densityMap * (len(dots) / np.sum(densityMap))

        image = image.transpose((2, 0, 1))

        if self.use_multiscale:
            multi_img = sample['scale_images']
            multi_img = multi_img.transpose((0, 3, 1, 2))

            sample['scale_images'] = multi_img.astype(np.float32)

        outdots = np.zeros((self.max_dot,2))

        count = len(dots)
        if count:
            outdots[:dots.shape[0],:] = dots
        sample['image'] = image.astype(np.float32)
        sample['densityMap'] = densityMap
        sample['dots'] = outdots
        sample['count'] = count
        sample.pop('sigma')
        return sample


class Normalize_t(object):
    def __init__(self,use_multiscale=False):
        self.use_multiscale = use_multiscale

    def __call__(self,sample):
        image = sample['image']
        image = (image - 127.5)/127.5
        sample['image'] = image
        if self.use_multiscale:
            multi_img = sample['scale_images']
            multi_img = (multi_img - 127.5) / 127.5

            sample['scale_images'] = multi_img

        return sample

class HeadCountDataset_trancos(Dataset):
    def __init__(self,max_iter,phase,  data_file, transform=None, use_mask=True, use_perspective=False, pmap_path=''):
        self.data_file = data_file
        f = open(self.data_file, 'r')
        self.data_idx = [i.strip() for i in f.readlines()]
        f.close()

        if phase == 'train':
            self.data_idx = self.data_idx * int(np.ceil(float(max_iter) / len(self.data_idx)))
        print('iteration length:', len(self.data_idx))

        self.transform = transform

        self.root_path = os.path.abspath(os.path.dirname(__file__) + os.path.sep + "../../../")

        self.use_perspective = use_perspective
        self.pmap_path = pmap_path
        self.use_mask = use_mask
        self.phase = phase
        print('use mask:', use_mask)

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):
        line = self.data_idx[idx]
        image_path = os.path.join('data/TRANCOS/images', line)
        dot_path = image_path.replace('.jpg', '.txt')
        if self.phase == 'train':
            image_path = image_path.replace('images', 'images_bilinear')
            dot_path = image_path.replace('images_bilinear', 'images').replace('.jpg', '.txt')

        image = io.imread(os.path.join(self.root_path, image_path))
       
        f = open(dot_path, 'r')
        notation = f.readlines()
        f.close()
        
        dots = [[int(i.strip().split()[0]),int(i.strip().split()[1])] for i in notation]
        dots = np.array(dots)
        if self.phase == 'train':
            dots = dots * 2
       
        idx = np.where(
            (dots[:, 0] >= 0) & (dots[:, 1] >= 0) & (dots[:, 0] < image.shape[1]) & (dots[:, 1] < image.shape[0]))
        dots = dots[idx][:, :2]
        if self.use_mask:
            mask_path = image_path.replace('images_bilinear', 'images').replace('.jpg', 'mask.mat')
            mask = scio.loadmat(mask_path)
            mask = mask['BW']
            if self.phase == 'train':
                mask_resize = sk_transform.resize(mask, (mask.shape[0]*2,mask.shape[1]*2),preserve_range=True)
            else:
                mask_resize = mask
           
            image[:,:,0] = image[:,:,0] * mask_resize
            image[:, :, 1] = image[:, :, 1] * mask_resize
            image[:, :, 2] = image[:, :, 2] * mask_resize
            dots = getMaskedDots(image, dots, mask_resize)
    
        if self.use_perspective:
            pers_file = h5py.File(self.pmap_path, 'r')
            pmap = np.array(pers_file['pmap'])

            pers_file.close()

            sigma = getPerspective(dots, pmap)

        else:
            sigma = torch.FloatTensor(len(dots)).fill_(10.0)
        sample = {'image': image, 'dots': dots, 'sigma': sigma,'image_name':line,'mask':mask_resize}
        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':

    headcount_dataset = HeadCountDataset_trancos(250000,'train','data/TRANCOS/image_sets/trainval.txt',pmap_path='',use_mask=True,
                                         transform=transforms.Compose(
                                             [IsColor_t(True), NineCrop_t(), RandomFlip_t(),Multiscale_t(cropscale=[0.8,0.6,0.4]),  PreferredSize_t(512,use_multiscale=True),ToTensor_t(use_multiscale=True), Normalize_t(use_multiscale=True)]))
    dataloader = DataLoader(headcount_dataset, batch_size=1, shuffle=False, num_workers=1)

    for i_batch, sample_batched in enumerate(dataloader):
        print('success')
        #showGt(sample_batched)
        showMultiscale(sample_batched)
       
