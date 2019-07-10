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
from src.lib.utils.image_opt import genDensity,showGt,getPerspective,getLevel,getAttentionDensity,showMultiscale,findSigma,getMaskedDots

class IsColor_m(object):
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


class RandomFlip_m(object):
    def __call__(self, sample):
        if random.random() < 0.5:
            image = sample['image']
            sample['image'] = sample['image'][:,::-1]
            if len(sample['dots']) != 0:
                sample['dots'][:,0] = image.shape[1] -1 - sample['dots'][:,0]
        return sample

class PreferredSize_m(object):
    def __init__(self,size = 0, use_multiscale=False):
        self.size = size
        self.use_multiscale=use_multiscale

    def __call__(self,sample):
        if self.size>0:
            image = sample['image']

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
                    h_s,w_s,c_s = img_s.shape
                    if h_s > w_s:
                        new_h_s, new_w_s = self.size, int(self.size * w_s / h_s + 0.5)
                        ratio_s = self.size / h_s
                    else:
                        new_h_s, new_w_s = int(self.size * h_s / w_s + 0.5), self.size
                        ratio_s = self.size / w_s

                    resized_img_s = sk_transform.resize(img_s,(new_h_s,new_w_s),preserve_range=True)
                    out_img_s = np.zeros((self.size,self.size,c_s),dtype=np.float32)
                    out_img_s[...] = 127.5
                    out_img_s[:new_h_s, :new_w_s, :] = resized_img_s
                    multi_img_new.append(out_img_s)

                sample['scale_images'] = np.array(multi_img_new)
            sample['dots'] = sample['dots']*ratio

            resized_image = sk_transform.resize(image,(new_h,new_w),preserve_range = True)
            out_image = np.zeros((self.size,self.size,c),dtype=np.float32)
            out_image[...] = 127.5
            out_image[:new_h,:new_w,:] = resized_image
            sample['image'] = out_image
        return sample


class NineCrop_m(object):
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
        if len(dots) != 0:
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

class Multiscale_m(object):
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

class ToTensor_m(object):
    def __init__(self,rescale=1.0,margin_size = 1001,max_dot = 4000,use_att=False,use_multiscale=False):
        self.rescale = rescale
        self.margin_size = margin_size
        self.max_dot = max_dot
        self.use_att = use_att
        self.use_multiscale = use_multiscale


    def __call__(self,sample):
        image = sample['image']
        
        dots = sample['dots']
        sigmas = sample['sigma']
        #sigmas = torch.FloatTensor(len(dots)).fill_(15.0)
        if self.use_att:
            densityMap = getAttentionDensity(image,3, dots, sigmas, self.margin_size,self.rescale)
        else:
            densityMap = genDensity(image, dots, sigmas, self.margin_size,self.rescale)
        
        if np.sum(densityMap)!= 0:
            densityMap = densityMap * (len(dots) / np.sum(densityMap))
        #densityMap = torch.FloatTensor(densityMap)
        
        image = image.transpose((2, 0, 1))
        if self.use_multiscale:
            multi_img = sample['scale_images']
            
            multi_img = multi_img.transpose((0,3,1,2))

            sample['scale_images'] = multi_img.astype(np.float32)


        outdots = np.zeros((self.max_dot,2))
        #outdots[:dots.shape[0],:] = dots
        count = len(dots)
        if count:
            outdots[:dots.shape[0],:] = dots
        sample['image'] = image.astype(np.float32)
        sample['densityMap'] = densityMap
        sample['dots'] = outdots
        sample['count'] = count
        sample.pop('sigma')
        return sample


class Normalize_m(object):
    def __init__(self,use_multiscale=False):
        self.use_multiscale=use_multiscale

    def __call__(self,sample):
        image = sample['image']
        image = (image - 127.5)/127.5
        sample['image'] = image
        if self.use_multiscale:
            multi_img = sample['scale_images']
            multi_img = (multi_img - 127.5)/127.5
            
            sample['scale_images'] = multi_img
        return sample

class HeadCountDataset_mall(Dataset):

    def __init__(self,max_iter,phase, data_file, transform=None,use_pers=True,use_mask=True,use_attention=False):
        self.data_file = data_file
        f = open(self.data_file,'r')
        self.data_idx = [i.strip() for i in f.readlines()]
        f.close()
        if phase == 'train':
            self.data_idx = self.data_idx * int(np.ceil(float(max_iter) / len(self.data_idx)))
        print('iteration length:', len(self.data_idx))
        self.transform = transform
        self.use_pmap = use_pers

        self.root_path = os.path.abspath(os.path.dirname(__file__)+os.path.sep+"../../../")
        self.use_att = use_attention
        self.use_mask = use_mask

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):
        line = self.data_idx[idx]
        img_path = line.strip()
        pmap_roi_path = 'data/Mall/perspective_roi.mat'
        
        image = io.imread(os.path.join(self.root_path, img_path)).astype(np.float32)
        frames = int(img_path.split('/')[-1][4:-4])
        
        notation = scio.loadmat(os.path.join(self.root_path, 'data/Mall/mall_gt.mat'), struct_as_record=False, squeeze_me=True)
        info = notation['frame'][frames-1]
        dots = info.loc
        idx = np.where((dots[:,0]>=0)&(dots[:,1]>=0)&(dots[:,0]<image.shape[1])&(dots[:,1]<image.shape[0]))
        dots = dots[idx]
        #sigma = findSigma(dots,3,0.3)
        # sigma = torch.FloatTensor(len(dots)).fill_(15)
        pmap_roi = scio.loadmat(os.path.join(self.root_path, pmap_roi_path), struct_as_record=False,
                                squeeze_me=True)
        if self.use_mask:
            mask = pmap_roi['roi'].mask

            #mask_resize = sk_transform.resize(mask, (mask.shape[0] * 4, mask.shape[1] * 4), preserve_range=True)
            # print('image shape', image.shape,mask_resize.shape)
            image[:, :, 0] = image[:, :, 0] * mask
            image[:, :, 1] = image[:, :, 1] * mask
            image[:, :, 2] = image[:, :, 2] * mask
            dots = getMaskedDots(image, dots, mask)
        if self.use_pmap:

            pmap = pmap_roi['pMapN']
            sigma = getPerspective(dots, pmap)

        else:
            sigma = findSigma(dots,3,0.3)


        if self.use_att:

            atv = getLevel(3, 0.1, np.array([3,9, 27]),dots,5)

            sample = {'image': image, 'dots': dots, 'sigma': atv, 'image_path': img_path}
        else:

            sample = {'image': image, 'dots': dots,'sigma':sigma, 'image_path': img_path}
        if self.transform:
            sample = self.transform(sample)
        
        return sample


if __name__ == '__main__':

    headcount_dataset = HeadCountDataset_mall(250000,'train','data/Mall/train.txt',use_pers=True,use_attention=False,
                                         transform=transforms.Compose(
                                             [IsColor_m(True),NineCrop_m(), RandomFlip_m(),Multiscale_m(cropscale=[0.75,0.5]),PreferredSize_m(512,use_multiscale=True), ToTensor_m(use_att=False,use_multiscale=True), Normalize_m(use_multiscale=True)]))
    dataloader = DataLoader(headcount_dataset, batch_size=1, shuffle=False, num_workers=1)

    for i_batch, sample_batched in enumerate(dataloader):
        print('success')
        #showGt(sample_batched)
        showMultiscale(sample_batched)
        
