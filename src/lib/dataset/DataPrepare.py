#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Willy

import matplotlib.pyplot as plt
import sys
import os
from PIL import Image
import scipy.io as scio
import cv2
import numpy as np
import math


def prepareShangHaiTech():
    DATASET_ROOT = 'data/ShanghaiTech'
    for part in ['part_A_final','part_B_final']:
        for phase in ['train_data','test_data']:
            DATASET_PATH = os.path.join(DATASET_ROOT,part,phase)
            fout = open(DATASET_PATH+'.txt','w+')
            for img_name in os.listdir(os.path.join(DATASET_PATH, 'images')):
                image_path = os.path.join(DATASET_PATH, 'images', img_name)
                gt_path = os.path.join(DATASET_PATH, 'ground_truth', 'GT_' + img_name.split('.')[0] + '.mat')
                fout.write(image_path + ' ' + gt_path + '\n')
            fout.close()
            
def prepareMall():
    set_path = 'data/Mall'
    train_f = open(os.path.join(set_path, 'train.txt'), 'w')
    for i in range(1, 801):
        file = 'seq_{:0>6}.jpg'.format(i)
        dot = 'dmap_{}.mat'.format(i)
        train_f.write(os.path.join(set_path, 'frames', file) + '\n')  # +' '+os.path.join(set_path,'gt',dot)

    train_f.closed
    test_f = open(os.path.join(set_path, 'test.txt'), 'w')
    for i in range(801, 2001):
        file = 'seq_{:0>6}.jpg'.format(i)
        test_f.write(os.path.join(set_path, 'frames', file) + '\n')  # +' '+os.path.join(set_path,'gt',dot)
    test_f.closed
    
def prepareUCSD():
    DATASET_ROOT = 'data/UCSD'
    #generate Maximal dataset
    #training
    training_f = open(os.path.join(DATASET_ROOT, 'imagedataset', 'train_maximal.txt'), 'w')
    for ix in range(605, 1400+5,5):
        xx = (ix - 1)//200
        yy = (ix - 200*xx)
        img_name = 'vidf1_33_00{}_f{:0>3}.png'.format(xx,yy)
        print(img_name)
        training_f.write( os.path.join(DATASET_ROOT,'images',img_name) + '\n' )
    training_f.close()

    testing_f = open(os.path.join(DATASET_ROOT, 'imagedataset', 'test_maximal.txt'), 'w')
    for ix in range(1,601):
        xx = (ix-1)//200
        yy = ix - 200*xx
        img_name = 'vidf1_33_00{}_f{:0>3}.png'.format(xx, yy)
        print(img_name)
        testing_f.write(os.path.join(DATASET_ROOT, 'images', img_name) + '\n')
    for ix in range(1401,2001):
        xx = (ix-1)//200
        yy = ix - 200*xx
        img_name = 'vidf1_33_00{}_f{:0>3}.png'.format(xx, yy)
        print(img_name)
        testing_f.write(os.path.join(DATASET_ROOT, 'images', img_name) + '\n')

    testing_f.close()

    #generate Downscale dataset
    # training
    training_f = open(os.path.join(DATASET_ROOT, 'imagedataset', 'train_downscale.txt'), 'w')
    for ix in range(1205, 1605, 5):
        xx = (ix - 1) // 200
        yy = (ix - 200 * xx)
        img_name = 'vidf1_33_00{}_f{:0>3}.png'.format(xx, yy)
        print(img_name)
        training_f.write(os.path.join(DATASET_ROOT, 'images', img_name) + '\n')
    training_f.close()

    testing_f = open(os.path.join(DATASET_ROOT, 'imagedataset', 'test_downscale.txt'), 'w')
    for ix in range(1, 1201):
        xx = (ix - 1) // 200
        yy = ix - 200 * xx
        img_name = 'vidf1_33_00{}_f{:0>3}.png'.format(xx, yy)
        print(img_name)
        testing_f.write(os.path.join(DATASET_ROOT, 'images', img_name) + '\n')
    for ix in range(1601, 2001):
        xx = (ix - 1) // 200
        yy = ix - 200 * xx
        img_name = 'vidf1_33_00{}_f{:0>3}.png'.format(xx, yy)
        print(img_name)
        testing_f.write(os.path.join(DATASET_ROOT, 'images', img_name) + '\n')

    testing_f.close()

    # generate Upscale dataset
    # training
    training_f = open(os.path.join(DATASET_ROOT, 'imagedataset', 'train_upscale.txt'), 'w')
    for ix in range(805, 1105, 5):
        xx = (ix - 1) // 200
        yy = (ix - 200 * xx)
        img_name = 'vidf1_33_00{}_f{:0>3}.png'.format(xx, yy)
        print(img_name)
        training_f.write(os.path.join(DATASET_ROOT, 'images', img_name) + '\n')
    training_f.close()

    testing_f = open(os.path.join(DATASET_ROOT, 'imagedataset', 'test_upscale.txt'), 'w')
    for ix in range(1, 801):
        xx = (ix - 1) // 200
        yy = ix - 200 * xx
        img_name = 'vidf1_33_00{}_f{:0>3}.png'.format(xx, yy)
        print(img_name)
        testing_f.write(os.path.join(DATASET_ROOT, 'images', img_name) + '\n')
    for ix in range(1101, 2001):
        xx = (ix - 1) // 200
        yy = ix - 200 * xx
        img_name = 'vidf1_33_00{}_f{:0>3}.png'.format(xx, yy)
        print(img_name)
        testing_f.write(os.path.join(DATASET_ROOT, 'images', img_name) + '\n')

    testing_f.close()

    # generate Minimal dataset
    # training
    training_f = open(os.path.join(DATASET_ROOT, 'imagedataset', 'train_minimal.txt'), 'w')
    for ix in range(640, 1360+80, 80):
        xx = (ix - 1) // 200
        yy = (ix - 200 * xx)
        img_name = 'vidf1_33_00{}_f{:0>3}.png'.format(xx, yy)
        print(img_name)
        training_f.write(os.path.join(DATASET_ROOT, 'images', img_name) + '\n')
    training_f.close()

    testing_f = open(os.path.join(DATASET_ROOT, 'imagedataset', 'test_minimal.txt'), 'w')
    for ix in range(1, 601):
        xx = (ix - 1) // 200
        yy = ix - 200 * xx
        img_name = 'vidf1_33_00{}_f{:0>3}.png'.format(xx, yy)
        print(img_name)
        testing_f.write(os.path.join(DATASET_ROOT, 'images', img_name) + '\n')
    for ix in range(1401, 2001):
        xx = (ix - 1) // 200
        yy = ix - 200 * xx
        img_name = 'vidf1_33_00{}_f{:0>3}.png'.format(xx, yy)
        print(img_name)
        testing_f.write(os.path.join(DATASET_ROOT, 'images', img_name) + '\n')

    testing_f.close()


    
def prepare_UCSD_gt():
    DATASET_ROOT = 'data/UCSD/vidf-cvpr'
    for i in range(10):
        notation_file = os.path.join(DATASET_ROOT, 'vidf1_33_00{}_frame_full.mat'.format(i))
        notation = scio.loadmat(notation_file, struct_as_record=False, squeeze_me=True)

        frames = notation['frame']
        for jx, frame in enumerate(frames):
            # Get dots
            loc = frame.loc
            print(loc.shape)
            scio.savemat(os.path.join('data/UCSD/ground-truth/vidf1_33_00{}_f{:0>3}.mat'.format(i,jx+1)), {'loc':loc})


    
def prepareVGGCell():
    DATASET_ROOT = 'data/VGGCell'

    test_file = os.path.join('data/test.txt')
    f = open(test_file, 'w')
    for img in os.listdir(DATASET_ROOT):
        print(img)
        if img[-8:-4] == 'cell' and img.split('.')[-1] == 'png':
            f.write(os.path.join(DATASET_ROOT,img) +'\n')

    f.close()

def prepareDublinCell():
    DATASET_ROOT = 'data/DublinCell'

    for phase in ['trainval','test']:
        DATASET_PATH = os.path.join(DATASET_ROOT,phase)
        fout = open(DATASET_PATH+'.txt','w+')
        for img_name in os.listdir(os.path.join(DATASET_PATH, 'images')):
            image_path = os.path.join(DATASET_PATH, 'images', img_name)
            #gt_path = os.path.join(DATASET_PATH, 'GT', img_name)
            fout.write(image_path + '\n')
        fout.close()

def prepareMBMCell():
    DATASET_ROOT = 'data/MBMCell'

    test_file = os.path.join('data/test.txt')
    f = open(test_file, 'w')
    for img in os.listdir(DATASET_ROOT):
        print(img)
        if img.split('_')[-2] != 'dots' and img.split('.')[-1] == 'png':
            f.write(os.path.join(DATASET_ROOT,img) +'\n')

    f.close()
    

        
nameFuncMapping = {'shanghai':prepareShangHaiTech}

#def Mirror()
def resize_bilinear(img,m, n):
    height, width, channels = img.shape
    emptyImage = np.zeros((m, n, channels), np.uint8)
    value = [0, 0, 0]
    sh = m / height
    sw = n / width
    for i in range(m):
        for j in range(n):
            x = i / sh
            y = j / sw
            p = (i + 0.0) / sh - x
            q = (j + 0.0) / sw - y
            x = int(x) - 1
            y = int(y) - 1
            for k in range(3):
                if x + 1 < m and y + 1 < n:
                    value[k] = int(
                        img[x, y][k] * (1 - p) * (1 - q) + img[x, y + 1][k] * q * (1 - p) + img[x + 1, y][k] * (
                        1 - q) * p + img[x + 1, y + 1][k] * p * q)
            emptyImage[i, j] = (value[0], value[1], value[2])
    return emptyImage

def S(x):
    x = np.abs(x)
    if 0 <= x < 1:
        return 1 - 2 * x * x + x * x * x
    if 1 <= x < 2:
        return 4 - 8 * x + 5 * x * x - x * x * x
    else:
        return 0

def resize_bicubic(img,m,n):
    height, width, channels = img.shape
    emptyImage = np.zeros((m, n, channels), np.uint8)
    sh = m / height
    sw = n / width
    for i in range(m):
        for j in range(n):
            x = i / sh
            y = j / sw
            p = (i + 0.0) / sh - x
            q = (j + 0.0) / sw - y
            x = int(x) - 2
            y = int(y) - 2
            A = np.array([
                [S(1 + p), S(p), S(1 - p), S(2 - p)]
            ])
            if x >= m - 3:
                m - 1
            if y >= n - 3:
                n - 1
            if x >= 1 and x <= (m - 3) and y >= 1 and y <= (n - 3):
                B = np.array([
                    [img[x - 1, y - 1], img[x - 1, y],
                     img[x - 1, y + 1],
                     img[x - 1, y + 1]],
                    [img[x, y - 1], img[x, y],
                     img[x, y + 1], img[x, y + 2]],
                    [img[x + 1, y - 1], img[x + 1, y],
                     img[x + 1, y + 1], img[x + 1, y + 2]],
                    [img[x + 2, y - 1], img[x + 2, y],
                     img[x + 2, y + 1], img[x + 2, y + 1]],

                ])
                C = np.array([
                    [S(1 + q)],
                    [S(q)],
                    [S(1 - q)],
                    [S(2 - q)]
                ])
                blue = np.dot(np.dot(A, B[:, :, 0]), C)[0, 0]
                green = np.dot(np.dot(A, B[:, :, 1]), C)[0, 0]
                red = np.dot(np.dot(A, B[:, :, 2]), C)[0, 0]

                # ajust the value to be in [0,255]
                def adjust(value):
                    if value > 255:
                        value = 255
                    elif value < 0:
                        value = 0
                    return value

                blue = adjust(blue)
                green = adjust(green)
                red = adjust(red)
                emptyImage[i, j] = np.array([blue, green, red], dtype=np.uint8)

    return emptyImage

def prepareUCSD_for_interpolation(type):
    DATASET_ROOT = 'data/UCSD'
    # type：
    # training
    training_f = open(os.path.join(DATASET_ROOT, 'interpolation_dataset', 'train_maximal_'+type+'.txt'), 'w')
    for ix in range(601, 1400):
        xx = (ix - 1) // 200
        yy = (ix - 200 * xx)
        img_name = 'vidf1_33_00{}_f{:0>3}.png'.format(xx, yy)
        print(img_name)
        training_f.write(os.path.join(DATASET_ROOT, 'images_for_interpolation', type, img_name) + '\n')
    training_f.close()

    testing_f = open(os.path.join(DATASET_ROOT, 'interpolation_dataset', 'test_maximal_'+type+'.txt'), 'w')
    for ix in range(1, 601):
        xx = (ix - 1) // 200
        yy = ix - 200 * xx
        img_name = 'vidf1_33_00{}_f{:0>3}.png'.format(xx, yy)
        print(img_name)
        testing_f.write(os.path.join(DATASET_ROOT, 'images_for_interpolation', type, img_name) + '\n')
    for ix in range(1401, 2001):
        xx = (ix - 1) // 200
        yy = ix - 200 * xx
        img_name = 'vidf1_33_00{}_f{:0>3}.png'.format(xx, yy)
        print(img_name)
        testing_f.write(os.path.join(DATASET_ROOT, 'images_for_interpolation', type, img_name) + '\n')

    testing_f.close()

def main(argv):
    nameFuncMapping[argv[0]]()
    return 0

if __name__ == '__main__':
    main(sys.argv[1:])
