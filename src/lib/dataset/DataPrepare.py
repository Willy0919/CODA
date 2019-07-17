#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Willy

import sys
import os
from PIL import Image


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

def main(argv):
    nameFuncMapping[argv[0]]()
    return 0

if __name__ == '__main__':
    main(sys.argv[1:])
